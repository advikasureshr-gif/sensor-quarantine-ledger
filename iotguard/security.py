"""
security.py - persistence, per-sensor quarantine and recovery, authentication, and a tamper-evident record.

    Ledger            append-only, hash-chained log of security and model events (SHA-256). It makes history
                      tamper-EVIDENT; it does not make a bad baseline correct (see the project notes).
    AdminAuth         PBKDF2-hashed admin secret with lockout. Calibration, release and recalibration need it.
    BaselineRegistry  baseline versions with SHA-256 digests, recorded in the ledger.
    SecurityManager   the "Persistence check" box: alert on a streak of SUSPICIOUS windows; quarantine a SENSOR
                      (not the system) after a sustained episode of suspicion attributed to it; probation and
                      admin-gated release.

What quarantine does: the sensor's features are masked out of scoring and learning (see GaussianModel), and
contradictions that depend on it stop freezing the model, so the healthy sensors keep operating. Nothing returns
to trust automatically: an admin must release the sensor, and it must then show a run of clean windows.

Attribution: each contradiction names the sensors it implicates (a pair relation such as door->PIR blames both
equally). A SUSPICIOUS window that is only "held" (no contradiction of its own) blames the sensor that contributes
most to the Mahalanobis score.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import DEFAULT_CONFIG, SecurityConfig
from .consensus import ConsensusResult, Decision, Trust
from .gaussian import BaselineSnapshot, GaussianModel

SENSORS = ("dht", "pir", "door", "ldr")
SENSOR_FEATURES = {"dht": ("temp_mean", "hum_mean"), "pir": ("pir_count", "pir_duty"),
                   "door": ("door_count",), "ldr": ("ldr_mean", "ldr_std")}
FEATURE_SENSOR = {f: s for s, fs in SENSOR_FEATURES.items() for f in fs}
EVIDENCE_SENSORS = {
    "door_pir_ok": ("door", "pir"), "door_pir_missing": ("door", "pir"),
    "door_light_ok": ("door", "ldr"), "door_light_missing": ("door", "ldr"), "light_unexplained": ("ldr",),
    "pir_unexplained": ("pir",), "pir_unexplained_rate": ("pir",), "pir_unverified": ("pir",), "pir_stuck": ("pir",),
    "temp_hum_inconsistent": ("dht",),
}


class AuthError(PermissionError):
    pass


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def _plain(x):
    """JSON-safe copy (numpy scalars/arrays become Python numbers/lists)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Enum):
        return x.value
    return x


class Ledger:
    """Append-only hash chain. Each record's hash covers the previous record's hash, so editing, deleting or
    reordering any record breaks every hash after it. With a `path`, records are also written as JSON lines."""
    GENESIS = "0" * 64

    def __init__(self, path=None, clock: Callable[[], str] = lambda: datetime.now().isoformat(timespec="seconds")) -> None:
        self.path = Path(path) if path else None
        self._clock = clock
        self.records: list[dict] = []
        if self.path is not None and self.path.exists() and self.path.stat().st_size > 0:
            self.records = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
            ok, bad = self.verify()
            if not ok:
                raise ValueError(f"ledger {self.path} is corrupted or has been tampered with (first bad record: {bad})")

    @staticmethod
    def _hash(prev: str, core: dict) -> str:
        return hashlib.sha256((prev + json.dumps(core, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()

    @property
    def head(self) -> str:
        return self.records[-1]["hash"] if self.records else self.GENESIS

    def append(self, event: str, t: Optional[float] = None, **data) -> dict:
        core = {"seq": len(self.records), "event": event, "t": t, "wall": self._clock(), "data": _plain(data), "prev": self.head}
        rec = {**core, "hash": self._hash(core["prev"], core)}
        self.records.append(rec)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
                f.flush()
        return rec

    def verify(self) -> tuple[bool, Optional[int]]:
        """(True, None) if the chain is intact, else (False, seq of the first bad record)."""
        prev = self.GENESIS
        for i, rec in enumerate(self.records):
            core = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("seq") != i or rec.get("prev") != prev or rec.get("hash") != self._hash(prev, core):
                return False, i
            prev = rec["hash"]
        return True, None

    def events(self, name: Optional[str] = None) -> list[dict]:
        return [r for r in self.records if name is None or r["event"] == name]


# ---------------------------------------------------------------------------
# admin authentication
# ---------------------------------------------------------------------------
class AdminAuth:
    """An admin secret stored only as a salted PBKDF2-SHA256 hash, with lockout after repeated failures."""

    def __init__(self, secret: Optional[str] = None, *, salt: Optional[bytes] = None, hash_hex: Optional[str] = None,
                 iterations: int = 200_000, max_failures: int = 3, lockout_s: float = 300.0,
                 ledger: Optional[Ledger] = None, clock: Callable[[], float] = time.monotonic) -> None:
        if (secret is None) == (hash_hex is None):
            raise ValueError("give either a secret (to create) or salt + hash_hex (to restore)")
        self.iterations, self.max_failures, self.lockout_s = iterations, max_failures, lockout_s
        self.ledger, self._clock = ledger, clock
        if secret is not None:
            if len(secret) < 8:
                raise ValueError("admin secret must be at least 8 characters")
            self.salt = secrets.token_bytes(16)
            self._hash = self._derive(secret)
        else:
            if salt is None:
                raise ValueError("salt is required to restore")
            self.salt, self._hash = salt, bytes.fromhex(hash_hex)
        self._failures = 0
        self._locked_until = 0.0

    def _derive(self, secret: str) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", secret.encode(), self.salt, self.iterations)

    def export(self) -> dict:
        """What may be stored on disk: never the secret."""
        return {"salt": self.salt.hex(), "hash_hex": self._hash.hex(), "iterations": self.iterations}

    @property
    def locked(self) -> bool:
        return self._clock() < self._locked_until

    def check(self, secret: Optional[str], t: Optional[float] = None) -> bool:
        now = self._clock()
        if now < self._locked_until:
            if self.ledger:
                self.ledger.append("auth_refused_locked", t)
            return False
        ok = secret is not None and hmac.compare_digest(self._derive(secret), self._hash)
        if ok:
            self._failures = 0
            return True
        self._failures += 1
        if self.ledger:
            self.ledger.append("auth_failed", t, failures=self._failures)
        if self._failures >= self.max_failures:
            self._locked_until = now + self.lockout_s
            self._failures = 0
            if self.ledger:
                self.ledger.append("auth_locked", t, seconds=self.lockout_s)
        return False

    def require(self, secret: Optional[str], t: Optional[float] = None) -> None:
        if not self.check(secret, t):
            raise AuthError("admin authentication failed" + (" (locked out)" if self.locked else ""))


# ---------------------------------------------------------------------------
# baseline versions
# ---------------------------------------------------------------------------
def baseline_digest(snap: BaselineSnapshot, version: int) -> str:
    """SHA-256 over the baseline's complete state and its version number."""
    h = hashlib.sha256()
    h.update(json.dumps({"names": list(snap.names), "version": version, "n_updates": snap.n_updates}, sort_keys=True).encode())
    for arr in (snap.mu, snap.sigma, snap.mu_cal, snap.sigma_cal):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


class BaselineRegistry:
    """Versions of the learned baseline. A new version is created by (re)calibration, never by routine learning;
    routine learning is covered by checkpoints (digests of the evolving state, recorded in the ledger)."""

    def __init__(self, ledger: Optional[Ledger] = None) -> None:
        self.ledger = ledger
        self.versions: list[dict] = []
        self._snapshots: dict[int, BaselineSnapshot] = {}

    @property
    def current(self) -> Optional[int]:
        active = [v["version"] for v in self.versions if v["active"]]
        return active[-1] if active else None

    def register(self, model: GaussianModel, reason: str, t: Optional[float] = None) -> int:
        """Record a new baseline as the active version (the previous one becomes inactive but is kept)."""
        for v in self.versions:
            if v["active"]:
                v["active"] = False
                if self.ledger:
                    self.ledger.append("baseline_retired", t, version=v["version"])
        version = len(self.versions) + 1
        snap = model.snapshot()
        digest = baseline_digest(snap, version)
        self.versions.append({"version": version, "digest": digest, "reason": reason, "t": t, "active": True})
        self._snapshots[version] = snap
        if self.ledger:
            self.ledger.append("baseline_created", t, version=version, digest=digest, reason=reason)
        return version

    def checkpoint(self, model: GaussianModel, t: Optional[float] = None) -> str:
        """Record a digest of the CURRENT learned state of the active version."""
        version = self.current
        if version is None:
            raise RuntimeError("no active baseline")
        digest = baseline_digest(model.snapshot(), version)
        if self.ledger:
            self.ledger.append("baseline_checkpoint", t, version=version, digest=digest, n_updates=model.n_updates)
        return digest

    def verify(self, version: int) -> bool:
        """Is the stored snapshot of `version` still exactly what was registered?"""
        rec = next((v for v in self.versions if v["version"] == version), None)
        return rec is not None and baseline_digest(self._snapshots[version], version) == rec["digest"]

    def snapshot(self, version: int) -> BaselineSnapshot:
        return self._snapshots[version]

    def get_state(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        """(versions as plain data, arrays keyed v<N>_mu ...) for saving."""
        arrays = {}
        for v, snap in self._snapshots.items():
            for field_name in ("mu", "sigma", "mu_cal", "sigma_cal"):
                arrays[f"reg{v}_{field_name}"] = getattr(snap, field_name)
        return [dict(v) for v in self.versions], arrays

    def set_state(self, versions: list[dict], arrays: dict[str, np.ndarray], names: tuple[str, ...], n_updates: dict[int, int]) -> None:
        """Restore, then check every stored snapshot still matches the digest recorded when it was registered."""
        self.versions = [dict(v) for v in versions]
        self._snapshots = {}
        for v in self.versions:
            n = v["version"]
            self._snapshots[n] = BaselineSnapshot(tuple(names), arrays[f"reg{n}_mu"].copy(), arrays[f"reg{n}_sigma"].copy(),
                                                  arrays[f"reg{n}_mu_cal"].copy(), arrays[f"reg{n}_sigma_cal"].copy(), int(n_updates[n]))
            if not self.verify(n):
                raise ValueError(f"baseline version {n} does not match its recorded digest: the saved state was altered")


# ---------------------------------------------------------------------------
# persistence, quarantine, recovery
# ---------------------------------------------------------------------------
class SensorState(str, Enum):
    ACTIVE = "ACTIVE"
    QUARANTINED = "QUARANTINED"
    PROBATION = "PROBATION"


@dataclass(frozen=True)
class SecurityEvent:
    kind: str                       # alert | sensor_quarantined | sensor_probation | sensor_recovered | probation_failed
    t: float
    window_index: int
    sensor: Optional[str]
    detail: str


class SecurityManager:
    def __init__(self, cfg: SecurityConfig = DEFAULT_CONFIG.security, window_s: float = DEFAULT_CONFIG.features.window_s,
                 ledger: Optional[Ledger] = None, auth: Optional[AdminAuth] = None) -> None:
        self.cfg, self.window_s, self.ledger, self.auth = cfg, window_s, ledger, auth
        self.state = {s: SensorState.ACTIVE for s in SENSORS}
        self._streak = 0
        self._alerted = False
        self._ep_start: dict[str, Optional[float]] = {s: None for s in SENSORS}
        self._ep_last: dict[str, int] = {s: -10**9 for s in SENSORS}
        self._clean_run = {s: 0 for s in SENSORS}
        self.events: list[SecurityEvent] = []
        self._pending_rebase: Optional[dict] = None

    # ---- saving and restoring: a restart must never clear a quarantine -----------------------------------------
    def get_state(self, t_ref: float, idx_ref: int) -> dict:
        far = -10 ** 8
        return {"states": {s: st.value for s, st in self.state.items()}, "streak": self._streak, "alerted": self._alerted,
                "clean_run": dict(self._clean_run),
                "ep_start_age": {s: (None if v is None else t_ref - v) for s, v in self._ep_start.items()},
                "ep_last_age": {s: (None if v < far else idx_ref - v) for s, v in self._ep_last.items()}}

    def set_state(self, d: dict) -> None:
        self.state = {s: SensorState(v) for s, v in d["states"].items()}
        self._streak, self._alerted, self._clean_run = int(d["streak"]), bool(d["alerted"]), {s: int(v) for s, v in d["clean_run"].items()}
        self._pending_rebase = d

    def _apply_rebase(self, t: float, idx: int) -> None:
        d, self._pending_rebase = self._pending_rebase, None
        for s in SENSORS:
            a = d["ep_start_age"].get(s)
            self._ep_start[s] = None if a is None else t - a
            b = d["ep_last_age"].get(s)
            self._ep_last[s] = -10 ** 9 if b is None else idx - b

    # ---- what quarantine changes for the rest of the pipeline ---------------------------------------------
    def muted_sensors(self) -> tuple[str, ...]:
        """Sensors that must not influence learning, scoring or the gate (quarantined, or on probation)."""
        return tuple(s for s, st in self.state.items() if st is not SensorState.ACTIVE)

    def muted_features(self) -> tuple[str, ...]:
        return tuple(f for s in self.muted_sensors() for f in SENSOR_FEATURES[s])

    def filter_consensus(self, cons: ConsensusResult) -> ConsensusResult:
        """Drop evidence that depends on a muted sensor (a pair relation needs both ends to be trusted)."""
        muted = set(self.muted_sensors())
        if not muted:
            return cons
        keep = tuple(e for e in cons.evidence if not (set(EVIDENCE_SENSORS.get(e[0], ())) & muted))
        pos = sum(v for _, v, _ in keep if v > 0)
        neg = sum(v for _, v, _ in keep if v < 0)
        return replace(cons, evidence=keep, pos=pos, neg=neg)

    # ---- per-window observation --------------------------------------------------------------------------
    @staticmethod
    def _blame(raw: ConsensusResult) -> dict[str, float]:
        b: dict[str, float] = {}
        for name, v, _ in raw.evidence:
            if v < 0:
                sensors = EVIDENCE_SENSORS.get(name, ())
                for s in sensors:
                    b[s] = b.get(s, 0.0) + 1.0 / len(sensors)
        return b

    def observe(self, t: float, decision: Decision, raw: ConsensusResult,
                contributions: Optional[dict[str, float]] = None) -> list[SecurityEvent]:
        """
        Feed one window's outcome. `decision` was made on the FILTERED evidence; `raw` is the unfiltered consensus
        (used for attribution and for watching sensors on probation); `contributions` = per-feature shares of D^2.
        Returns the security events this window caused.
        """
        c, out, idx = self.cfg, [], decision.index
        if self._pending_rebase is not None:
            self._apply_rebase(t, idx)
        suspicious = decision.label is Trust.SUSPICIOUS

        # --- persistence: a streak of SUSPICIOUS windows raises one alert per episode
        self._streak = self._streak + 1 if suspicious else 0
        if self._streak >= c.suspicious_streak_for_alert and not self._alerted:
            self._alerted = True
            out.append(self._emit("alert", t, idx, None, f"{self._streak} consecutive SUSPICIOUS windows: {decision.reason}"))
        if not suspicious:
            self._alerted = False

        # --- attribution (only SUSPICIOUS windows build episodes; probation watches every window)
        raw_blame = self._blame(raw)
        blame: dict[str, float] = {}
        if suspicious:
            blame = {s: w for s, w in raw_blame.items() if self.state[s] is SensorState.ACTIVE}
            if not blame and contributions:
                top: dict[str, float] = {}
                for f, share in contributions.items():
                    s = FEATURE_SENSOR.get(f)
                    if s and self.state[s] is SensorState.ACTIVE and share > 0:
                        top[s] = top.get(s, 0.0) + share
                if top:
                    blame = {max(top, key=top.get): 1.0}

        # --- episodes and quarantine
        for s in SENSORS:
            if self.state[s] is not SensorState.ACTIVE:
                continue
            if blame.get(s, 0.0) > 0:
                if self._ep_start[s] is None or idx - self._ep_last[s] > c.episode_gap_windows:
                    self._ep_start[s] = t
                self._ep_last[s] = idx
                if t - self._ep_start[s] + self.window_s >= c.quarantine_after_s:
                    self.state[s] = SensorState.QUARANTINED
                    self._clean_run[s] = 0
                    out.append(self._emit("sensor_quarantined", t, idx, s,
                                          f"suspicion attributed to the {s} for {(t - self._ep_start[s]) / 3600:.1f} h"))
            elif self._ep_start[s] is not None and idx - self._ep_last[s] > c.episode_gap_windows:
                self._ep_start[s] = None

        # --- probation: consecutive clean windows restore trust; any blame sends the sensor back
        for s in SENSORS:
            if self.state[s] is not SensorState.PROBATION:
                continue
            if raw_blame.get(s, 0.0) > 0:
                self.state[s] = SensorState.QUARANTINED
                self._clean_run[s] = 0
                out.append(self._emit("probation_failed", t, idx, s, "contradictions involving the sensor reappeared"))
            else:
                self._clean_run[s] += 1
                if self._clean_run[s] >= c.probation_windows:
                    self.state[s] = SensorState.ACTIVE
                    self._ep_start[s], self._clean_run[s] = None, 0
                    out.append(self._emit("sensor_recovered", t, idx, s, f"{c.probation_windows} consecutive clean windows"))
        return out

    # ---- admin actions -----------------------------------------------------------------------------------
    def admin_release(self, sensor: str, credential: Optional[str], t: float = 0.0, window_index: int = -1) -> SecurityEvent:
        """An admin has inspected a quarantined sensor: start its probation. Requires the admin credential."""
        if sensor not in SENSORS:
            raise ValueError(f"unknown sensor {sensor!r}")
        if self.auth is not None:
            self.auth.require(credential, t)
        if self.state[sensor] is not SensorState.QUARANTINED:
            raise ValueError(f"{sensor} is {self.state[sensor].value}, not quarantined")
        self.state[sensor] = SensorState.PROBATION
        self._clean_run[sensor] = 0
        return self._emit("sensor_probation", t, window_index, sensor, "released by admin; on probation")

    def _emit(self, kind: str, t: float, idx: int, sensor: Optional[str], detail: str) -> SecurityEvent:
        ev = SecurityEvent(kind, t, idx, sensor, detail)
        self.events.append(ev)
        if self.ledger:
            self.ledger.append(kind, t, window_index=idx, sensor=sensor, detail=detail)
        return ev