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

import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .calibration import CalibrationResult, CalibrationStatus, Calibrator
from .config import DEFAULT_CONFIG, Config, ConsensusConfig
from .consensus import ConsensusEngine, ConsensusResult, Decision, Gate, Trust
from .contract import Reading
from .features import WindowExtractor, WindowFeatures
from .gaussian import BaselineSnapshot, GaussianModel
from .preprocess import Preprocessor
from .security import AdminAuth, BaselineRegistry, Ledger, SecurityEvent, SecurityManager
from .watchdog import HealthConfig, Watchdog


class StateIntegrityError(ValueError):
    """The saved state does not match the hash the ledger recorded when it was saved: it was altered."""


class StateMismatchError(ValueError):
    """The saved state was made with different settings (features, thresholds, ...) than the ones in use now."""


def config_fingerprint(cfg: Config, consensus: ConsensusConfig) -> str:
    """Identifies the settings a saved state depends on; a state is never loaded under different ones."""
    return hashlib.sha256(repr((cfg.features, cfg.online_stats, cfg.anomaly, cfg.calibration, cfg.security, consensus)).encode()).hexdigest()


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
    naive_drift: float = math.nan                 # the always-learning twin's drift (only with naive_twin=True)
    divergence: float = math.nan                  # distance between the gated baseline and that twin (calibration covariance)
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
                 checkpoint_every_windows: int = 2880, state_path=None, autosave_every_windows: int = 120,
                 wall_clock=time.time, health: Optional[HealthConfig] = None) -> None:
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
        self.state_path = Path(state_path) if state_path else None
        self.autosave_every = autosave_every_windows
        self._wall = wall_clock
        self._since_save = 0
        self._last_t: Optional[float] = None
        self._last_index: Optional[int] = None
        self.watchdog = Watchdog(health if health is not None else HealthConfig())     # reports only; never changes a decision
        self.health_events: list[SecurityEvent] = []
        self._pending_health: list[SecurityEvent] = []

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
        for h in self.watchdog.observe(reading):
            ev = SecurityEvent(h.kind, h.t, self.n_windows, h.sensor, h.detail)
            self.health_events.append(ev)
            self._pending_health.append(ev)
            if self.ledger:
                self.ledger.append(h.kind, h.t, fault=h.fault, sensor=h.sensor, detail=h.detail)
        out = []
        for w in self.extractor.push(self.pre.process(reading)):
            out.append(self.process_window(w))
        if self._pending_health and out:                # shown with the next finished window
            out[0].events[:0] = self._pending_health
            self._pending_health = []
        return out

    def run(self, source: Iterable[Reading], on_outcome: Optional[Callable[[WindowOutcome], None]] = None) -> list[WindowOutcome]:
        """Process a whole source (live sources run until stopped). Returns every outcome unless `on_outcome` is given."""
        results = []
        try:
            for r in source:
                for o in self.push(r):
                    if on_outcome:
                        on_outcome(o)
                    else:
                        results.append(o)
        finally:
            if self.state_path is not None and self._last_t is not None:
                self.save()                       # also on Ctrl+C or a lost connection
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
        self._last_t, self._last_index = t, cons.index
        if self.phase is Phase.CALIBRATING:
            self._calibrate(o, lite if lite is not None else _Lite(valid, t, x, index), cons)
        elif self.phase is Phase.OPERATING and valid:
            self._operate(o, x, cons, t)
        self._autosave()
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
            dv = self.model.mu - self.naive.mu
            o.naive_drift = self.naive.drift_distance()
            o.divergence = float(math.sqrt(max(dv @ self.model._P_cal @ dv, 0.0)))
        o.score, o.decision, o.alpha, o.muted = score, decision, alpha, self.security.muted_sensors()
        o.drift = self.model.drift_distance()
        self._since_checkpoint += 1
        if self.ledger is not None and self._since_checkpoint >= self.checkpoint_every:
            self._since_checkpoint = 0
            self.registry.checkpoint(self.model, t)

    def _autosave(self) -> None:
        if self.state_path is not None:
            self._since_save += 1
            if self._since_save >= self.autosave_every:
                self.save()

    # ---- saving and restoring ---------------------------------------------------------------------------------
    def save(self, path=None) -> str:
        """
        Write everything a restart must not lose (see the module docs of the parts) to one file, atomically, and put its
        SHA-256 in the ledger so a later edit of the file is detectable. Returns the file's hash.
        """
        path = Path(path or self.state_path)
        if path is None:
            raise ValueError("no path to save to")
        t_ref = 0.0 if self._last_t is None else self._last_t
        idx_ref = 0 if self._last_index is None else self._last_index
        versions, arrays = self.registry.get_state()
        meta = {"format": 1, "saved_wall": self._wall(), "fingerprint": config_fingerprint(self.cfg, self.consensus_cfg),
                "phase": self.phase.value, "n_windows": self.n_windows, "since_checkpoint": self._since_checkpoint,
                "naive_wanted": self._naive_wanted, "registry": versions,
                "registry_updates": {str(v): self.registry.snapshot(v).n_updates for v in self.registry._snapshots},
                "engine": self.engine.get_state(), "gate": self.gate.get_state(),
                "security": self.security.get_state(t_ref, idx_ref), "has_model": self.model is not None,
                "has_naive": self.naive is not None and self.model is not None}
        for tag, model in (("model", self.model), ("naive", self.naive if self.model is not None else None)):
            if model is not None and model.fitted:
                snap = model.snapshot()
                meta[f"{tag}_names"], meta[f"{tag}_updates"] = list(snap.names), snap.n_updates
                for f in ("mu", "sigma", "mu_cal", "sigma_cal"):
                    arrays[f"{tag}_{f}"] = getattr(snap, f)
        if self.calibrator is not None and self.phase is Phase.CALIBRATING:
            cm, X = self.calibrator.get_state()
            meta["calibrator"] = cm
            if X is not None:
                arrays["calib_x"] = X
        arrays["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                                   # a crash mid-save leaves the previous file intact
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.ledger is not None:
            self.ledger.append("state_saved", t_ref, sha256=digest, file=path.name)
        self._since_save = 0
        return digest

    @classmethod
    def load(cls, path, cfg: Config = DEFAULT_CONFIG, consensus: Optional[ConsensusConfig] = None, ledger: Optional[Ledger] = None,
             auth: Optional[AdminAuth] = None, max_downtime_s: float = 600.0, wall_now: Optional[float] = None, **kw) -> "Pipeline":
        """
        Resume from a saved state. Refuses (StateIntegrityError) if the ledger recorded a different hash for the last
        save, and (StateMismatchError) if the settings differ from those the state was made with. If the host was off
        longer than `max_downtime_s`, the belief about who is in the room is reset (it is stale), but the baseline, the
        room's learned settings and every sensor's quarantine are kept.
        """
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if ledger is not None:
            saves = ledger.events("state_saved")
            if saves and saves[-1]["data"]["sha256"] != digest:
                raise StateIntegrityError("the saved state does not match the hash recorded in the ledger: it was altered or is not the latest save")
        z = np.load(io.BytesIO(data), allow_pickle=False)
        meta = json.loads(bytes(z["meta_json"]).decode())
        consensus_cfg = consensus if consensus is not None else cfg.consensus
        if meta["fingerprint"] != config_fingerprint(cfg, consensus_cfg):
            raise StateMismatchError("the saved state was made with different settings; load it with the settings it was saved under")
        p = cls(cfg, consensus=consensus_cfg, ledger=ledger, auth=auth, naive_twin=meta["naive_wanted"], **kw)
        arrays = {k: z[k] for k in z.files}
        reg_names = tuple(meta["model_names"]) if meta["has_model"] else GaussianModel(cfg.online_stats, cfg.features.feature_names).names
        p.registry.set_state(meta["registry"], {k: v for k, v in arrays.items() if k.startswith("reg")}, reg_names,
                             {int(k): v for k, v in meta["registry_updates"].items()})
        for tag, attr in (("model", "model"), ("naive", "naive")):
            if meta.get(f"has_{tag}") and f"{tag}_mu" in arrays:
                m = GaussianModel(cfg.online_stats, cfg.features.feature_names)
                m.restore(BaselineSnapshot(tuple(meta[f"{tag}_names"]), arrays[f"{tag}_mu"], arrays[f"{tag}_sigma"],
                                           arrays[f"{tag}_mu_cal"], arrays[f"{tag}_sigma_cal"], meta[f"{tag}_updates"]))
                setattr(p, attr, m)
        p.phase = Phase(meta["phase"])
        p.n_windows, p._since_checkpoint = meta["n_windows"], meta["since_checkpoint"]
        p.engine.set_state(meta["engine"])
        p.gate.set_state(meta["gate"])
        p.security.set_state(meta["security"])
        if "calibrator" in meta:
            p.calibrator = Calibrator(cfg.calibration, cfg.online_stats, consensus_cfg, cfg.features.feature_names, p.window_s)
            p.calibrator.set_state(meta["calibrator"], arrays.get("calib_x"))
        now = wall_now if wall_now is not None else p._wall()
        downtime = max(0.0, now - meta["saved_wall"])
        if downtime > max_downtime_s:
            p.engine.reset_occupancy()
            p.gate.reset()
        if ledger is not None:
            ledger.append("state_loaded", None, sha256=digest, downtime_s=round(downtime, 1), occupancy_reset=downtime > max_downtime_s)
        return p