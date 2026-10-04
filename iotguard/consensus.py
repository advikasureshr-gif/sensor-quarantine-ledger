"""
consensus.py - cross-sensor consensus and the trust gate: the heart of the project.

    WindowFeatures --ConsensusEngine--> ConsensusResult --Gate(+ Mahalanobis score D)--> Decision

ConsensusEngine looks only at fixed physical relationships between sensors; it never uses the
learned mu/Sigma. It is stateful (it tracks who is believed to be in the room) and must see
consecutive windows in order.

Occupancy tracking
------------------
    UNKNOWN   at start. First PIR activity -> OCCUPIED; silence for `vacancy_timeout_s` -> EMPTY.
    EMPTY     a door opening followed by PIR motion (an "entry") -> OCCUPIED.
    OCCUPIED  a further such passage is exit-like (an exit or a visitor); if the PIR then stays
              silent for `vacancy_timeout_s` the room is believed EMPTY. Motion shortly after the
              passage means someone stayed, so the exit suspicion is dropped.
Motion while OCCUPIED is normal. Motion while EMPTY with no door opening to explain it is the
signature of a spoofed or stuck PIR, a masked door, or a replay.

Evidence (each is positive or negative, see ConsensusConfig for the thresholds)
    door_pir_ok / door_pir_missing           door opening vs motion
    door_light_ok / door_light_missing        door opening vs corridor-light step
    pir_unexplained                           motion in an empty room, no door opening
    pir_stuck                                 PIR continuously high far too long
    temp_hum_inconsistent                     abrupt temperature step without a humidity response
    light_unexplained                         abrupt light step in an empty room, no door opening

Gate
----
Combines the evidence with the Mahalanobis score D:
    SUSPICIOUS  negative evidence >= suspicious_neg                         -> freeze (alpha = 0)
    UNCERTAIN   weaker contradiction, or an unexplained statistical outlier  -> slow learning
    TRUSTED     no contradiction and D within threshold, OR a corroborated event
                (positive evidence) with a not-absurd D                      -> full learning
A corroborated event may legitimately score high (a real door entry has D around 8), so
corroboration overrides a high D; a contradiction overrides everything.
Suspicion memory: contradictions raise a decaying level; while it is high, windows that are
still anomalous stay SUSPICIOUS instead of being slowly learned, until the anomaly ends or
`suspicion_max_hold_windows` pass.

Known limit, by design: an attack that keeps every checked relationship plausible produces no
contradiction and is treated like a legitimate change (UNCERTAIN, slow learning).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Optional

from .config import DEFAULT_CONFIG, ConsensusConfig, FeatureConfig, OnlineStatsConfig
from .features import WindowFeatures


class Trust(str, Enum):
    TRUSTED = "TRUSTED"
    UNCERTAIN = "UNCERTAIN"
    SUSPICIOUS = "SUSPICIOUS"


class Occupancy(str, Enum):
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


@dataclass(frozen=True)
class ConsensusResult:
    index: int
    occupancy: Occupancy                        # believed state at the end of the window
    evidence: tuple[tuple[str, float, str], ...]   # (name, signed value, human-readable detail)
    pos: float                                  # sum of positive evidence
    neg: float                                  # sum of negative evidence (<= 0)

    @property
    def contradictions(self) -> tuple[str, ...]:
        return tuple(n for n, v, _ in self.evidence if v < 0)

    @property
    def corroborations(self) -> tuple[str, ...]:
        return tuple(n for n, v, _ in self.evidence if v > 0)


@dataclass(frozen=True)
class Decision:
    index: int
    label: Trust
    reason: str
    d: float
    suspicion: float                            # gate's suspicion level AFTER this window
    occupancy: Occupancy

    def alpha(self, cfg: OnlineStatsConfig = DEFAULT_CONFIG.online_stats) -> float:
        """The learning rate this decision allows."""
        return {Trust.TRUSTED: cfg.alpha_trusted, Trust.UNCERTAIN: cfg.alpha_uncertain,
                Trust.SUSPICIOUS: cfg.alpha_suspicious}[self.label]


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
class ConsensusEngine:
    HIST_S = 40.0          # seconds of PIR/LDR history kept for look-back checks

    def __init__(self, cfg: ConsensusConfig = DEFAULT_CONFIG.consensus,
                 features_cfg: FeatureConfig = DEFAULT_CONFIG.features,
                 sample_period_s: float = DEFAULT_CONFIG.sampling.sample_period_s) -> None:
        if cfg.light_check not in ("off", "positive_only", "full"):
            raise ValueError("light_check must be 'off', 'positive_only' or 'full'")
        if features_cfg.hop_s != features_cfg.window_s:
            raise ValueError("ConsensusEngine requires disjoint windows (hop_s == window_s)")
        self.cfg = cfg
        self._dt = sample_period_s
        self.reset()

    def reset(self) -> None:
        self.state = Occupancy.UNKNOWN
        self._first_t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._last_motion_t: Optional[float] = None
        self._exit_candidate = False
        self._passage_t = -math.inf
        self._hist: deque[tuple[float, int, int]] = deque()      # (t, pir, ldr_raw)
        self._opens: deque[float] = deque()                       # recent confirmed door openings
        self._pending_doors: list[float] = []
        self._pending_steps: list[float] = []
        self._pir_prev = 0
        self._pir_start = 0.0
        self._pir_explained = False
        self._prev_ldr: Optional[int] = None
        self._prev_th: Optional[tuple[float, float, int]] = None  # (temp, hum, window index)

    # ---- helpers -------------------------------------------------------------------------
    def _door_explains(self, start: float, now: float) -> bool:
        c = self.cfg
        return any(start - 2.0 <= te <= min(now, start + c.door_pir_post_s) for te in self._opens)

    def _passage(self, te: float) -> None:
        if self.state in (Occupancy.UNKNOWN, Occupancy.EMPTY):      # an entry
            self.state = Occupancy.OCCUPIED
            self._exit_candidate = False
        else:                                                        # exit-like (leaving or a visitor)
            self._exit_candidate = True
            self._passage_t = te

    def _light_step(self, te: float) -> Optional[bool]:
        before = [l for tt, _, l in self._hist if te - 4.0 <= tt <= te - 1.5]
        after = [l for tt, _, l in self._hist if te <= tt <= te + 3.0]
        if len(before) < 2 or len(after) < 2:
            return None
        return median(after) - median(before) >= self.cfg.door_light_step_adc

    def _temp_hum(self, w: WindowFeatures) -> Optional[str]:
        c = self.cfg
        if "temp_mean" not in w.names or "hum_mean" not in w.names:
            return None
        t, h = w.feature("temp_mean"), w.feature("hum_mean")
        if not (w.valid and not w.dht_carried and math.isfinite(t) and math.isfinite(h)):
            return None
        out = None
        prev = self._prev_th
        if prev is not None and w.index == prev[2] + 1:
            dT, dH = t - prev[0], h - prev[1]
            if abs(dT) >= c.temp_step_c or abs(dH) >= c.hum_step_pct:
                lo, hi = c.hum_per_c_range
                ratio = (-dH / dT) if dT != 0 else math.inf
                if not lo <= ratio <= hi:
                    out = f"temperature moved {dT:+.1f} C while humidity moved {dH:+.1f} %RH"
        self._prev_th = (t, h, w.index)
        return out

    # ---- one window ---------------------------------------------------------------------------
    def evaluate(self, w: WindowFeatures) -> ConsensusResult:
        if not w.samples:
            raise ValueError("window has no samples; extract windows with keep_samples=True")
        c = self.cfg
        n_ok = n_bad = n_step = n_nostep = n_unexplained_steps = 0
        unexplained_s = 0.0

        for s in w.samples:
            t = s.t
            dt = min(max(t - self._prev_t, 0.0), 2 * self._dt) if self._prev_t is not None else self._dt
            if self._first_t is None:
                self._first_t = t
            if s.door_event == 1:
                self._opens.append(t)
                self._pending_doors.append(t)

            # --- PIR bookkeeping
            if s.pir:
                if not self._pir_prev:                                # a new high interval starts
                    self._pir_start = t - s.pir_high_s
                    self._pir_explained = False
                if self.state is Occupancy.UNKNOWN:
                    self.state = Occupancy.OCCUPIED
                if not self._pir_explained:
                    self._pir_explained = self.state is Occupancy.OCCUPIED or self._door_explains(self._pir_start, t)
                if not self._pir_explained:
                    unexplained_s += dt
                self._last_motion_t = t
                if self._exit_candidate and t - self._passage_t >= c.exit_clear_s:
                    self._exit_candidate = False                       # someone stayed inside
            self._pir_prev = s.pir

            # --- abrupt light step
            if self._prev_ldr is not None and abs(s.ldr_raw - self._prev_ldr) >= c.abrupt_light_step_adc:
                self._pending_steps.append(t)
            self._prev_ldr = s.ldr_raw

            self._hist.append((t, s.pir, s.ldr_raw))
            while self._hist and self._hist[0][0] < t - self.HIST_S:
                self._hist.popleft()
            while self._opens and self._opens[0] < t - 60.0:
                self._opens.popleft()

            # --- judge door openings once their look-ahead is complete
            while self._pending_doors and self._pending_doors[0] + c.door_pir_post_s <= t:
                te = self._pending_doors.pop(0)
                if any(p for tt, p, _ in self._hist if te - c.door_pir_pre_s <= tt <= te + c.door_pir_post_s):
                    n_ok += 1
                    self._passage(te)
                else:
                    n_bad += 1
                step = self._light_step(te)
                if step is True:
                    n_step += 1
                elif step is False:
                    n_nostep += 1

            # --- judge abrupt light steps
            while self._pending_steps and self._pending_steps[0] + c.abrupt_light_post_s <= t:
                ts = self._pending_steps.pop(0)
                if self.state is Occupancy.EMPTY and not any(ts - 6.0 <= te <= ts + c.abrupt_light_post_s for te in self._opens):
                    n_unexplained_steps += 1

            # --- vacancy
            if self.state is Occupancy.OCCUPIED and self._exit_candidate:
                ref = max(self._last_motion_t if self._last_motion_t is not None else -math.inf, self._passage_t)
                if t - ref >= c.vacancy_timeout_s:
                    self.state, self._exit_candidate = Occupancy.EMPTY, False
            elif self.state is Occupancy.UNKNOWN and t - self._first_t >= c.vacancy_timeout_s:
                self.state = Occupancy.EMPTY
            self._prev_t = t

        # --- assemble evidence
        ev: list[tuple[str, float, str]] = []
        if n_ok:
            ev.append(("door_pir_ok", c.w_door_pir * min(n_ok, 2), f"{n_ok} door opening(s) followed by motion"))
        if n_bad:
            ev.append(("door_pir_missing", -c.w_door_pir * min(n_bad, 2), f"{n_bad} door opening(s) with no motion"))
        if c.light_check != "off":
            if n_step:
                ev.append(("door_light_ok", c.w_light * min(n_step, 2), f"{n_step} opening(s) with a light step"))
            if n_nostep and c.light_check == "full":
                ev.append(("door_light_missing", -c.w_light * min(n_nostep, 2), f"{n_nostep} opening(s) with no light step"))
        if unexplained_s >= c.pir_unexplained_min_s:
            ev.append(("pir_unexplained", -c.w_pir_unexplained,
                       f"{unexplained_s:.1f} s of motion with nobody believed inside and no door opening"))
        if w.pir_high_max_s >= c.pir_max_continuous_s:
            ev.append(("pir_stuck", -c.w_pir_stuck, f"PIR continuously high for {w.pir_high_max_s:.0f} s"))
        if n_unexplained_steps:
            ev.append(("light_unexplained", -c.w_abrupt_light, f"{n_unexplained_steps} abrupt light step(s) in an empty room"))
        th = self._temp_hum(w)
        if th:
            ev.append(("temp_hum_inconsistent", -c.w_temp_hum, th))

        pos = sum(v for _, v, _ in ev if v > 0)
        neg = sum(v for _, v, _ in ev if v < 0)
        return ConsensusResult(w.index, self.state, tuple(ev), pos, neg)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
class Gate:
    def __init__(self, cfg: ConsensusConfig = DEFAULT_CONFIG.consensus,
                 d_threshold: float = DEFAULT_CONFIG.anomaly.mahalanobis_threshold) -> None:
        self.cfg = cfg
        self.d_threshold = d_threshold
        self.suspicion = 0.0
        self._held_run = 0

    def reset(self) -> None:
        self.suspicion, self._held_run = 0.0, 0

    def decide(self, cons: ConsensusResult, d: float) -> Decision:
        c = self.cfg
        anomalous = d > self.d_threshold
        self.suspicion *= c.suspicion_decay
        if self.suspicion < c.suspicion_hold:
            self._held_run = 0

        # 1. classify from the evidence
        if cons.neg <= -c.suspicious_neg:
            label, reason = Trust.SUSPICIOUS, "contradiction: " + ", ".join(cons.contradictions)
        elif cons.neg < 0:
            label, reason = Trust.UNCERTAIN, "weak contradiction: " + ", ".join(cons.contradictions)
        elif not anomalous:
            label, reason = Trust.TRUSTED, "consistent" if not cons.corroborations else "consistent, corroborated"
        elif cons.pos >= c.corroborated_min and d <= c.corroborated_max_d:
            label, reason = Trust.TRUSTED, "corroborated event: " + ", ".join(cons.corroborations)
        else:
            label, reason = Trust.UNCERTAIN, "unexplained statistical outlier"

        # 2. suspicion memory
        if label is Trust.SUSPICIOUS:
            self.suspicion = min(c.suspicion_cap, self.suspicion + 1.0)
            self._held_run = 0
        elif label is Trust.UNCERTAIN and anomalous and self.suspicion >= c.suspicion_hold:
            if self._held_run < c.suspicion_max_hold_windows:
                label, reason = Trust.SUSPICIOUS, "held: anomaly persists after an earlier contradiction"
                self.suspicion = min(c.suspicion_cap, self.suspicion + c.suspicion_refresh)
                self._held_run += 1
            else:
                reason += " (suspicion hold expired)"
                self.suspicion = 0.0
        return Decision(cons.index, label, reason, d, self.suspicion, cons.occupancy)