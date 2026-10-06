"""
pipeline.py - the whole system as one streaming object (the complete flow diagram).

    Reading --Preprocessor--> CleanReading --WindowExtractor--> window
        --ConsensusEngine--> evidence
        phase CALIBRATING:  Calibrator (screening) -> baseline v1 (registered, hashed) -> OPERATING
        phase OPERATING:    Mahalanobis score (quarantined sensors masked) -> Gate (trusted / uncertain / suspicious)
                            -> learning rate (full / slow / freeze) -> model update
                            -> SecurityManager (persistence alerts, sensor quarantine, probation) -> Ledger

Use `push(reading)` for live data, `run(source)` for any Source, or `process(...)` to drive the core from
pre-computed windows (this is how the experiment code evaluates it). All three share the same core step.

Phases: IDLE (waiting for an authenticated start_calibration) -> CALIBRATING -> OPERATING, or
CALIBRATION_FAILED if the data never met the requirements (no baseline is built from weak data).
Recalibration (a new calibration while operating) also needs the admin credential and creates baseline v2;
v1 is kept in the registry as inactive history.

Admin-gated actions: start_calibration, release_sensor. If no AdminAuth is configured they are open, which is
only appropriate for simulation and development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np

from .calibration import CalibrationResult, CalibrationStatus, Calibrator
from .config import DEFAULT_CONFIG, Config, ConsensusConfig
from .consensus import ConsensusEngine, ConsensusResult, Decision, Gate, Trust
from .contract import Reading
from .features import WindowExtractor, WindowFeatures
from .gaussian import GaussianModel
from .preprocess import Preprocessor
from .security import AdminAuth, BaselineRegistry, Ledger, SecurityEvent, SecurityManager


class Phase(str, Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    OPERATING = "OPERATING"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"


@dataclass
class WindowOutcome:
    index: int
    t_start: float
    phase: Phase
    valid: bool
    consensus: Optional[ConsensusResult] = None
    score: float = math.nan                       # Mahalanobis distance (muted sensors left out)
    decision: Optional[Decision] = None
    alpha: float = 0.0                            # learning rate applied to this window
    drift: float = math.nan                       # baseline drift since calibration
    muted: tuple[str, ...] = ()                   # sensors quarantined or on probation
    events: list[SecurityEvent] = field(default_factory=list)
    calibration: Optional[CalibrationResult] = None   # set on the window that finished (or failed) calibration
    features: Optional[np.ndarray] = None

    def line(self) -> str:
        """One human-readable line, for a terminal."""
        base = f"t={self.t_start:9.1f}s {self.phase.value:<11}"
        if self.phase is Phase.OPERATING and self.decision is not None:
            muted = f" muted={','.join(self.muted)}" if self.muted else ""
            occ = f" room={self.consensus.occupancy.value}" if self.consensus else ""
            return (f"{base} D={self.score:5.2f} {self.decision.label.value:<10} alpha={self.alpha:<7g}{occ}{muted}  {self.decision.reason}")
        if self.calibration is not None:
            return f"{base} {self.calibration.summary()}"
        return base


@dataclass
class _Lite:
    """What the calibrator needs from a window, so the core can be driven without a WindowFeatures."""
    valid: bool
    t_start: float
    x: np.ndarray
    index: int = 0


class Pipeline:
    def __init__(self, cfg: Config = DEFAULT_CONFIG, consensus: Optional[ConsensusConfig] = None,
                 ledger: Optional[Ledger] = None, auth: Optional[AdminAuth] = None, naive_twin: bool = False,
                 checkpoint_every_windows: int = 2880) -> None:
        self.cfg = cfg
        self.consensus_cfg = consensus if consensus is not None else cfg.consensus
        self.ledger, self.auth = ledger, auth
        if auth is not None and ledger is not None and auth.ledger is None:
            auth.ledger = ledger                      # failed logins and lockouts must be on the record
        self.window_s = cfg.features.window_s
        self.pre = Preprocessor(cfg.preprocess)
        self.extractor = WindowExtractor(cfg.features, cfg.sampling.sample_period_s, keep_samples=True)
        self.engine = ConsensusEngine(self.consensus_cfg, cfg.features, cfg.sampling.sample_period_s)
        self.gate = Gate(self.consensus_cfg, cfg.anomaly.mahalanobis_threshold)
        self.security = SecurityManager(cfg.security, self.window_s, ledger, auth)
        self.registry = BaselineRegistry(ledger)
        self.calibrator: Optional[Calibrator] = None
        self.model: Optional[GaussianModel] = None
        self.naive: Optional[GaussianModel] = GaussianModel(cfg.online_stats, cfg.features.feature_names) if naive_twin else None
        self._naive_wanted = naive_twin
        self.phase = Phase.IDLE
        self.checkpoint_every = checkpoint_every_windows
        self._since_checkpoint = 0
        self.n_windows = 0

    # ---- admin actions ----------------------------------------------------------------------------------
    def _authorize(self, credential: Optional[str], t: float) -> None:
        if self.auth is not None:
            self.auth.require(credential, t)

    def start_calibration(self, credential: Optional[str] = None, t: float = 0.0) -> None:
        """Begin (or restart) calibration. Needs the admin credential. While calibrating, no decisions are made."""
        self._authorize(credential, t)
        recal = self.model is not None
        self.engine = ConsensusEngine(self.consensus_cfg, self.cfg.features, self.cfg.sampling.sample_period_s)   # re-learn this room
        self.calibrator = Calibrator(self.cfg.calibration, self.cfg.online_stats, self.consensus_cfg,
                                     self.cfg.features.feature_names, self.window_s)
        self.phase = Phase.CALIBRATING
        if self.ledger:
            self.ledger.append("recalibration_started" if recal else "calibration_started", t)

    def adopt_baseline(self, model: GaussianModel, reason: str = "adopted", t: float = 0.0, engine_calibrated: bool = True) -> int:
        """Skip calibration and operate with an existing baseline (used by offline evaluation and for loading a saved one)."""
        self.model = model
        if self._naive_wanted:
            self.naive = model.copy()
        if engine_calibrated:
            self.engine.end_calibration()
        self.phase = Phase.OPERATING
        return self.registry.register(model, reason, t)

    def release_sensor(self, sensor: str, credential: Optional[str] = None, t: float = 0.0) -> SecurityEvent:
        return self.security.admin_release(sensor, credential, t)

    # ---- streaming -----------------------------------------------------------------------------------------
    def push(self, reading: Reading) -> list[WindowOutcome]:
        """Feed one raw reading; returns an outcome for every window it completed."""
        out = []
        for w in self.extractor.push(self.pre.process(reading)):
            out.append(self.process_window(w))
        return out

    def run(self, source: Iterable[Reading], on_outcome: Optional[Callable[[WindowOutcome], None]] = None) -> list[WindowOutcome]:
        """Process a whole source (live sources run until stopped). Returns every outcome unless `on_outcome` is given."""
        results = []
        for r in source:
            for o in self.push(r):
                if on_outcome:
                    on_outcome(o)
                else:
                    results.append(o)
        return results

    def process_window(self, w: WindowFeatures) -> WindowOutcome:
        cons = self.engine.evaluate(w) if w.samples else None
        if cons is None:
            raise ValueError("windows must carry their samples (keep_samples=True) for the consensus engine")
        return self.process(w.x, cons, w.t_start, w.index, w.valid, lite=w)

    # ---- the core step (shared by live use and offline evaluation) -------------------------------------------
    def process(self, x: np.ndarray, cons: ConsensusResult, t: float, index: int, valid: bool = True, lite=None) -> WindowOutcome:
        o = WindowOutcome(index, t, self.phase, valid, cons, features=x)
        self.n_windows += 1
        if self.phase is Phase.CALIBRATING:
            self._calibrate(o, lite if lite is not None else _Lite(valid, t, x, index), cons)
        elif self.phase is Phase.OPERATING and valid:
            self._operate(o, x, cons, t)
        return o

    def _calibrate(self, o: WindowOutcome, w, cons: ConsensusResult) -> None:
        res = self.calibrator.add(w, cons)
        if res is None:
            return
        o.calibration = res
        if res.status is CalibrationStatus.READY:
            self.model = res.model
            if self._naive_wanted:
                self.naive = res.model.copy()
            self.engine.end_calibration()
            version = self.registry.register(res.model, "calibration", o.t_start)
            if self.ledger:
                self.ledger.append("calibration_finished", o.t_start, version=version, trusted=res.n_trusted, candidates=res.n_candidates,
                                   rejected=res.rejected, span_s=res.span_s)
            self.phase = Phase.OPERATING
        else:
            self.phase = Phase.CALIBRATION_FAILED
            ev = SecurityEvent("calibration_failed", o.t_start, o.index, None, res.reason)
            o.events.append(ev)
            self.security.events.append(ev)
            if self.ledger:
                self.ledger.append("calibration_failed", o.t_start, reason=res.reason, trusted=res.n_trusted, candidates=res.n_candidates)

    def _operate(self, o: WindowOutcome, x: np.ndarray, raw: ConsensusResult, t: float) -> None:
        muted_feats = self.security.muted_features()
        score = self.model.score(x, exclude=muted_feats)
        decision = self.gate.decide(self.security.filter_consensus(raw), score)
        alpha = decision.alpha(self.cfg.online_stats)
        contributions = None
        if decision.label is Trust.SUSPICIOUS and not any(v < 0 for _, v, _ in raw.evidence):
            contributions = self.model.contributions(x, exclude=muted_feats)       # for attributing a held window
        o.events.extend(self.security.observe(t, decision, raw, contributions))
        self.model.update(x, alpha, exclude=muted_feats)
        if self.naive is not None:
            self.naive.update(x, self.cfg.online_stats.alpha_trusted)
        o.score, o.decision, o.alpha, o.muted = score, decision, alpha, self.security.muted_sensors()
        o.drift = self.model.drift_distance()
        self._since_checkpoint += 1
        if self.ledger is not None and self._since_checkpoint >= self.checkpoint_every:
            self._since_checkpoint = 0
            self.registry.checkpoint(self.model, t)