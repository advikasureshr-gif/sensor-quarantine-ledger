"""
calibration.py - screened construction of baseline v1 (the "INITIALIZATION" branch of the flow).

    candidate windows --validate--> trusted windows --enough?--> fit mu/Sigma --> baseline v1
                          |                               |
              reject: consensus contradiction   no: keep collecting; at the time limit -> FAILED + alert
              reject: robust statistical outlier (unless corroborated by a door entry)

The Calibrator does not decide WHEN calibration may start; the pipeline does that, behind admin
authentication (security.py). It only judges the data it is given.

Why screen at all: calibration is the one moment an attacker (or a fault) can write the baseline directly.
Windows that contradict the physics of the room, or are extreme for no corroborated reason, must not become
"normal". Why door-corroborated windows are kept even when extreme: a real entry puts a corridor-light step and
a PIR burst in one window (we measured D around 8), and a baseline that excluded all of them would call every
entry abnormal.

The data requirements (duration, trusted count, trusted share) exist so that a weak baseline is never built
just because time ran out: if they are not met by `max_duration_s`, calibration FAILS and no baseline exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .config import DEFAULT_CONFIG, CalibrationConfig, ConsensusConfig, OnlineStatsConfig
from .consensus import ConsensusResult, Gate, Trust
from .features import WindowFeatures
from .gaussian import GaussianModel


class CalibrationStatus(str, Enum):
    COLLECTING = "COLLECTING"       # requirements not met yet; keep feeding windows
    READY = "READY"                 # requirements met; a fitted model is available
    FAILED = "FAILED"               # time limit reached without meeting the requirements; no baseline


@dataclass
class CalibrationResult:
    status: CalibrationStatus
    reason: str
    model: Optional[GaussianModel]
    n_candidates: int
    n_trusted: int
    rejected: dict[str, int] = field(default_factory=dict)      # reason -> count
    span_s: float = 0.0
    trusted_mask: Optional[np.ndarray] = None                   # over the candidate windows, in arrival order

    @property
    def trusted_fraction(self) -> float:
        return self.n_trusted / self.n_candidates if self.n_candidates else 0.0

    def summary(self) -> str:
        rej = ", ".join(f"{k} {v}" for k, v in self.rejected.items() if v) or "none"
        return (f"{self.status.value}: {self.n_trusted} of {self.n_candidates} windows trusted ({self.trusted_fraction:.0%}) "
                f"over {self.span_s / 3600:.1f} h; rejected: {rej}. {self.reason}")


class Calibrator:
    def __init__(self, cfg: CalibrationConfig = DEFAULT_CONFIG.calibration,
                 online: OnlineStatsConfig = DEFAULT_CONFIG.online_stats,
                 consensus: ConsensusConfig = DEFAULT_CONFIG.consensus,
                 feature_names=DEFAULT_CONFIG.features.feature_names,
                 window_s: float = DEFAULT_CONFIG.features.window_s) -> None:
        self.cfg, self.online, self.consensus, self.feature_names, self.window_s = cfg, online, consensus, tuple(feature_names), window_s
        self._probe = GaussianModel(online, self.feature_names)       # supplies the model's transform and variance floors
        self._gate = Gate(consensus)
        self._t: list[float] = []
        self._x: list[np.ndarray] = []
        self._bad: list[Optional[str]] = []                           # consensus rejection reason, or None
        self._corroborated: list[bool] = []
        self.n_invalid = 0
        self.result: Optional[CalibrationResult] = None
        self._pending_shift = False

    # ---- saving and restoring: a restart during a 2-day calibration must not throw the collected data away ----------
    def get_state(self) -> tuple[dict, Optional[np.ndarray]]:
        """(plain data, feature matrix). Times are stored relative to the newest candidate (<= 0)."""
        last = self._t[-1] if self._t else 0.0
        meta = {"t_rel": [t - last for t in self._t], "bad": list(self._bad), "corroborated": [bool(c) for c in self._corroborated],
                "n_invalid": self.n_invalid}
        return meta, (np.vstack(self._x) if self._x else None)

    def set_state(self, meta: dict, X: Optional[np.ndarray]) -> None:
        self.reset()
        self._t = list(meta["t_rel"])
        self._x = [] if X is None else [row.copy() for row in X]
        self._bad, self._corroborated, self.n_invalid = list(meta["bad"]), list(meta["corroborated"]), int(meta["n_invalid"])
        self._pending_shift = bool(self._t)

    # ---- feeding -------------------------------------------------------------------------------
    def add(self, w: WindowFeatures, cons: ConsensusResult) -> Optional[CalibrationResult]:
        """
        Offer one window. Returns a result when calibration has just finished (READY or FAILED), else None.
        Once finished, further windows are ignored.
        """
        if self.result is not None and self.result.status is not CalibrationStatus.COLLECTING:
            return None
        if self._pending_shift:                       # first window after a restore: anchor the saved times to the new clock
            base = w.t_start - self.window_s
            self._t = [base + r for r in self._t]
            self._pending_shift = False
        if not w.valid:
            self.n_invalid += 1
            return None
        verdict, why = self._gate._negative_verdict(cons)
        self._t.append(w.t_start)
        self._x.append(w.x)
        self._bad.append(why if verdict in (Trust.SUSPICIOUS, Trust.UNCERTAIN) else None)
        self._corroborated.append(cons.pos >= self.consensus.corroborated_min and cons.neg == 0)
        n = len(self._t)
        if n % self.cfg.check_every_windows == 0 or self.span_s >= self.cfg.max_duration_s:
            res = self.evaluate()
            if res.status is not CalibrationStatus.COLLECTING:
                return res
        return None

    @property
    def span_s(self) -> float:
        return (self._t[-1] - self._t[0] + self.window_s) if self._t else 0.0

    @property
    def n_candidates(self) -> int:
        return len(self._t)

    # ---- screening and the requirements check -------------------------------------------------------
    def screen(self) -> tuple[np.ndarray, dict[str, int]]:
        """Which candidate windows are trusted (bool array in arrival order), and why the others were rejected."""
        n = len(self._t)
        if n == 0:
            return np.zeros(0, dtype=bool), {}
        passes = np.array([b is None for b in self._bad])
        rejected = {"consensus contradiction": int((~passes).sum()), "statistical outlier": 0}
        trusted = passes.copy()
        if passes.sum() >= 10:
            Z = self._probe._transform(np.vstack(self._x))
            floor = np.sqrt(self._probe._floor2)
            med = np.median(Z[passes], axis=0)
            mad = np.median(np.abs(Z[passes] - med), axis=0)
            scale = np.maximum(1.4826 * mad, floor)
            zmax = (np.abs(Z - med) / scale).max(axis=1)
            outlier = (zmax > self.cfg.mad_outlier_threshold) & ~np.array(self._corroborated)
            trusted &= ~outlier
            rejected["statistical outlier"] = int((passes & outlier).sum())
        return trusted, rejected

    def evaluate(self) -> CalibrationResult:
        """Screen what has been collected and decide: READY (model fitted), COLLECTING, or FAILED."""
        cfg = self.cfg
        trusted, rejected = self.screen()
        n, k = len(trusted), int(trusted.sum())
        span = self.span_s
        frac = k / n if n else 0.0
        missing = []
        if span < cfg.min_duration_s:
            missing.append(f"only {span / 3600:.1f} h of data (need {cfg.min_duration_s / 3600:.1f} h)")
        if k < cfg.min_trusted_windows:
            missing.append(f"only {k} trusted windows (need {cfg.min_trusted_windows})")
        if n and frac < cfg.min_trusted_fraction:
            missing.append(f"only {frac:.0%} of windows are trusted (need {cfg.min_trusted_fraction:.0%}): the data looks contaminated or the room unstable")
        if not missing:
            model = GaussianModel(self.online, self.feature_names)
            model.fit(np.vstack([x for x, keep in zip(self._x, trusted) if keep]))
            self.result = CalibrationResult(CalibrationStatus.READY, "baseline can be built", model, n, k, rejected, span, trusted)
        elif span >= cfg.max_duration_s:
            self.result = CalibrationResult(CalibrationStatus.FAILED, "time limit reached: " + "; ".join(missing), None, n, k, rejected, span, trusted)
        else:
            self.result = CalibrationResult(CalibrationStatus.COLLECTING, "; ".join(missing), None, n, k, rejected, span, trusted)
        return self.result

    def reset(self) -> None:
        """Discard everything collected (a new calibration attempt)."""
        self._t, self._x, self._bad, self._corroborated = [], [], [], []
        self.n_invalid = 0
        self.result = None
        self._pending_shift = False