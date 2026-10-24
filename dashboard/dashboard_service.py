"""
dashboard_service.py - the ONLY place the dashboard touches the IoTGuard engine.

It builds a scenario with the engine's own `make_source`, feeds the readings (labels stripped, exactly as the engine's
own run.py does) into the engine's own `Pipeline`, and records what the pipeline returns. Nothing here scores,
classifies or thresholds anything itself. Evaluation (precision/recall/...) is computed AFTER the run from the
pipeline's decisions and the scenario's own ground-truth manifest, and only when that ground truth exists.

Design rules
  * one run = one fresh Pipeline + Ledger (no state shared between runs)
  * only one run at a time (RunManager); a second start is refused, never queued silently
  * runs execute in a background thread so Streamlit reruns cannot start them twice
  * results are saved as JSON/CSV only (no pickle) under outputs/runs/<run_id>/
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                       # make `import iotguard` resolve to the copy shipped beside this file
    sys.path.insert(0, str(ROOT))

from iotguard.pipeline import Phase, Pipeline                     # noqa: E402
from iotguard.presets import room_setup                           # noqa: E402
from iotguard.scenarios import PROFILES, SCENARIOS, make_source, profile_configs   # noqa: E402
from iotguard.security import Ledger                              # noqa: E402
from iotguard.sources import strip_labels                         # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "runs"
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
MAX_DAYS = 14.0
FEATURES = ("temp_mean", "hum_mean", "pir_count", "pir_duty", "door_count", "ldr_mean", "ldr_std")
WINDOW_COLUMNS = ["index", "t_start", "t_end", "phase", "valid", "decision", "reason", "score", "alpha", "drift", "suspicion",
                  "occupancy", "occ_conf", "evidence_pos", "evidence_neg", "evidence", "muted", *FEATURES,
                  "truth", "attack_tags"]
EVENT_COLUMNS = ["kind", "t", "window_index", "sensor", "detail"]


class ParamError(ValueError):
    """The requested run parameters are invalid."""


class RunBusyError(RuntimeError):
    """A run is already in progress."""


class RunCancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def json_safe(x: Any) -> Any:
    """Plain JSON-compatible copy: numpy -> python, Enum -> value, inf/nan -> None, tuples -> lists."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = dataclasses.asdict(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def min_days(scenario: str) -> float:
    """Shortest run (days, rounded up to 0.5) that still contains every scripted attack, from the scenario's own definition."""
    sc = SCENARIOS[scenario]
    b = sc.build()                                           # default length; start times come from the scenario itself
    last = max([a.start_s for a in b.attacks] + [c.start_s for c in b.changes] + [b.calibration_s]) / 86400.0
    return min(sc.days, math.ceil((last + 0.25) * 2) / 2)


def scenario_catalog() -> list[dict]:
    """The scenarios that really exist in iotguard/scenarios.py (nothing is added here)."""
    return [{"name": s.name, "title": s.title, "experiment": s.experiment, "expect": s.expect,
             "default_days": s.days, "min_days": min_days(s.name), "calibration_days": s.calibration_days,
             "contaminates_calibration": s.contaminates_calibration} for s in SCENARIOS.values()]


def preview_schedule(params: RunParams) -> dict:
    """Scripted attacks / legitimate changes for the chosen parameters, taken from the scenario definition (no run needed)."""
    sc = SCENARIOS[params.scenario]
    b = sc.build(profile_configs(params.profile)[0], params.days, params.strength)
    return json_safe({"calibration_s": b.calibration_s, "duration_s": b.duration_s,
                      "attacks": [{"tag": a.tag, "kind": a.kind, "channel": a.channel, "start_s": a.start_s, "end_s": a.end_s,
                                   "magnitude": a.magnitude, "ramp_s": a.ramp_s, "peak": a.peak} for a in b.attacks],
                      "legit_changes": [{"kind": c.kind, "start_s": c.start_s, "label_end_s": c.label_end_s, "magnitude": c.magnitude} for c in b.changes]})


def profile_names() -> list[str]:
    return list(PROFILES)


# ---------------------------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunParams:
    scenario: str
    seed: int = 0
    days: Optional[float] = None
    strength: float = 1.0
    profile: str = "default"


def validate_params(scenario: Any, seed: Any = 0, days: Any = None, strength: Any = 1.0, profile: Any = "default") -> RunParams:
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ParamError(f"unknown scenario {scenario!r}; available: {', '.join(SCENARIOS)}")
    sc = SCENARIOS[scenario]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ParamError("seed must be a whole number")
    seed = int(seed)
    if not 0 <= seed <= 2**31 - 1:
        raise ParamError("seed must be between 0 and 2147483647")
    if days is not None:
        if isinstance(days, bool) or not isinstance(days, (int, float, np.number)) or not math.isfinite(float(days)):
            raise ParamError("days must be a finite number")
        days = float(days)
        if days > MAX_DAYS:
            raise ParamError(f"days must be at most {MAX_DAYS:g} (longer runs take too long on a laptop)")
        if days <= sc.calibration_days:
            raise ParamError(f"days must be greater than the scenario's {sc.calibration_days:g}-day calibration period")
    if isinstance(strength, bool) or not isinstance(strength, (int, float, np.number)) or not math.isfinite(float(strength)):
        raise ParamError("strength must be a finite number")
    strength = float(strength)
    if not 0.1 <= strength <= 3.0:
        raise ParamError("strength must be between 0.1 and 3.0")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ParamError(f"unknown profile {profile!r}; available: {', '.join(PROFILES)}")
    try:                                                    # the engine's own consistency checks (attack inside the run, etc.)
        sc.build(profile_configs(profile)[0], days, strength)
    except ValueError as e:
        raise ParamError(str(e)) from e
    return RunParams(scenario, seed, days, strength, profile)


# ---------------------------------------------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class RunRecord:
    run_id: str
    params: dict
    status: str = "queued"                  # queued | running | completed | failed | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: float = 0.0
    error: Optional[str] = None
    rows: list = field(default_factory=list)
    events: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _df_n: int = field(default=-1, repr=False)

    @property
    def windows(self) -> pd.DataFrame:
        n = len(self.rows)
        if self._df is None or self._df_n != n:
            self._df = pd.DataFrame(list(self.rows[:n]), columns=WINDOW_COLUMNS)
            self._df_n = n
        return self._df

    @property
    def events_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.events), columns=EVENT_COLUMNS)

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.run_id} | {p.get('scenario')} | seed {p.get('seed')}"


def ground_truth_for_window(manifest: dict, t_start: float, t_end: float) -> tuple[str, str]:
    """'attack' if the window overlaps a scenario attack interval, else 'legit_change' if it overlaps a labelled
    legitimate change, else 'normal'. Uses only the scenario's own manifest."""
    tags = []
    for a in manifest.get("attacks", []):
        end = a.get("end_s")
        if a["start_s"] < t_end and (end is None or end > t_start):
            tags.append(a["tag"])
    if tags:
        return "attack", ",".join(sorted(set(tags)))
    for c in manifest.get("legit_changes", []):
        end = c.get("label_end_s")
        if c["start_s"] < t_end and (end is None or end > t_start):
            return "legit_change", ""
    return "normal", ""


def _row(o, window_s: float, manifest: dict) -> dict:
    d, c = o.decision, o.consensus
    feats = list(o.features) if o.features is not None else [None] * len(FEATURES)
    truth, tags = ground_truth_for_window(manifest, o.t_start, o.t_start + window_s) if manifest else ("unknown", "")
    row = {"index": o.index, "t_start": o.t_start, "t_end": o.t_start + window_s, "phase": o.phase.value, "valid": bool(o.valid),
           "decision": d.label.value if d else None, "reason": d.reason if d else None,
           "score": _f(o.score), "alpha": _f(o.alpha) if d else None, "drift": _f(o.drift),
           "suspicion": _f(d.suspicion) if d else None,
           "occupancy": (d.occupancy.value if d else (c.occupancy.value if c else None)),
           "occ_conf": _f(c.occ_conf) if c else None,
           "evidence_pos": _f(c.pos) if c else None, "evidence_neg": _f(c.neg) if c else None,
           "evidence": json.dumps([[n, _f(v), det] for n, v, det in c.evidence]) if c else "[]",
           "muted": ",".join(o.muted), "truth": truth, "attack_tags": tags}
    for name, v in zip(FEATURES, feats):
        row[name] = _f(v)
    return row


# ---------------------------------------------------------------------------------------------------------------
# evaluation (only with real ground truth; the unit is one 30 s window)
# ---------------------------------------------------------------------------------------------------------------
def _ratio(a: float, b: float) -> Optional[float]:
    return a / b if b > 0 else None


def evaluate(df: pd.DataFrame, manifest: dict) -> dict:
    """
    Window-level evaluation of the OPERATING phase against the scenario manifest. Positive = window overlaps a scripted
    attack. "Flagged" = decision SUSPICIOUS (strict). UNCERTAIN is not counted as flagged; the looser reading
    (SUSPICIOUS or UNCERTAIN) is reported separately. Metrics are None when their denominator is zero.
    Windows after an attack ends are negatives, so the gate's deliberate hold after an attack shows up as false positives.
    """
    out: dict = {"available": False, "unit": "30 s window (OPERATING phase only)", "reason": None}
    if not manifest or df.empty:
        out["reason"] = "no scenario ground truth or no windows"
        return out
    op = df[(df["phase"] == Phase.OPERATING.value) & df["decision"].notna() & df["valid"]]
    if op.empty:
        out["reason"] = "the pipeline never reached the OPERATING phase, so there are no decisions to score"
        return out
    pos = op["truth"] == "attack"
    out["available"] = True
    out["n_eval_windows"] = int(len(op))
    out["n_attack_windows"] = int(pos.sum())
    out["n_legit_change_windows"] = int((op["truth"] == "legit_change").sum())
    for tag, mask in (("strict", op["decision"] == "SUSPICIOUS"), ("loose", op["decision"].isin(["SUSPICIOUS", "UNCERTAIN"]))):
        tp, fp = int((mask & pos).sum()), int((mask & ~pos).sum())
        fn, tn = int((~mask & pos).sum()), int((~mask & ~pos).sum())
        p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = _ratio(2 * p * r, p + r) if p is not None and r is not None else None
        out[tag] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "fpr": _ratio(fp, fp + tn)}
    return out


def detection_latency(df: pd.DataFrame, manifest: dict) -> dict:
    """Seconds from the first scripted attack start (inside the operating phase) to the end of the first SUSPICIOUS window
    that overlaps an attack. Only the first attack is measured; multi-attack scenarios are not averaged."""
    res = {"available": False, "reason": None}
    attacks = [a for a in manifest.get("attacks", []) if a["start_s"] >= manifest.get("calibration_s", 0.0)]
    if not attacks:
        res["reason"] = "scenario has no attack in the operating phase"
        return res
    first = min(attacks, key=lambda a: a["start_s"])
    res.update(attack_tag=first["tag"], attack_start_s=first["start_s"])
    if df.empty:
        res["reason"] = "no windows"
        return res
    hit = df[(df["decision"] == "SUSPICIOUS") & (df["t_end"] > first["start_s"]) & (df["truth"] == "attack")]
    res["available"] = True
    if hit.empty:
        res.update(detected=False, latency_s=None)
    else:
        res.update(detected=True, latency_s=float(hit.iloc[0]["t_end"] - first["start_s"]), first_flag_t_end=float(hit.iloc[0]["t_end"]))
    return res


def verify_records(records: list[dict]) -> tuple[bool, Optional[int]]:
    """Re-check a hash chain with the engine's own Ledger.verify()."""
    led = Ledger()
    led.records = [dict(r) for r in records]
    return led.verify()


# ---------------------------------------------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------------------------------------------
def execute_run(rec: RunRecord) -> None:
    """Run one scenario through the real pipeline, filling `rec` as it goes. Never raises: failures go into rec.error."""
    rec.status, rec.started = "running", time.time()
    try:
        p = RunParams(**rec.params)
        cfg, cons = room_setup()
        world_cfg, sensors = profile_configs(p.profile)
        src = make_source(p.scenario, p.seed, days=p.days, world_cfg=world_cfg, sensors=sensors, strength=p.strength)
        rec.manifest = json_safe(src.manifest())
        rec.config = json_safe({"window_s": cfg.features.window_s, "sample_period_s": cfg.sampling.sample_period_s,
                                "anomaly": cfg.anomaly, "online_stats": cfg.online_stats, "security": cfg.security,
                                "calibration": cfg.calibration, "consensus": cons, "preset": "room_setup (validated on simulator per source docs)"})
        pipe = Pipeline(cfg, consensus=cons, ledger=Ledger())
        pipe.start_calibration()
        window_s, duration = cfg.features.window_s, src.built.duration_s
        calib_note: dict = {}

        def stream():
            for r in strip_labels(src):                    # the pipeline never sees ground truth
                if rec.cancel.is_set():
                    raise RunCancelled()
                yield r

        def on_outcome(o):
            rec.rows.append(_row(o, window_s, rec.manifest))
            for e in o.events:
                rec.events.append({"kind": e.kind, "t": e.t, "window_index": e.window_index, "sensor": e.sensor, "detail": e.detail})
            if o.calibration is not None:
                c = o.calibration
                calib_note.update(status=getattr(getattr(c, "status", None), "value", str(getattr(c, "status", ""))),
                                  reason=getattr(c, "reason", None), n_trusted=getattr(c, "n_trusted", None),
                                  n_candidates=getattr(c, "n_candidates", None), span_s=getattr(c, "span_s", None),
                                  summary=c.summary())
            rec.progress = min(1.0, (o.t_start + window_s) / duration)

        try:
            pipe.run(stream(), on_outcome=on_outcome)
            rec.status = "completed"
        except RunCancelled:
            rec.status = "cancelled"
        for h in getattr(pipe, "_pending_health", []):      # health events not yet attached to a window
            rec.events.append({"kind": h.kind, "t": h.t, "window_index": h.window_index, "sensor": h.sensor, "detail": h.detail})
        rec.ledger = json_safe(pipe.ledger.records)
        ok, bad = pipe.ledger.verify()
        df = rec.windows
        op = df[(df["phase"] == "OPERATING") & df["decision"].notna()]
        counts = {k: int((op["decision"] == k).sum()) for k in ("TRUSTED", "UNCERTAIN", "SUSPICIOUS")}
        ev_kinds = pd.Series([e["kind"] for e in rec.events], dtype=str).value_counts().to_dict() if rec.events else {}
        last = op.iloc[-1] if len(op) else None
        rec.summary = json_safe({
            "partial": rec.status != "completed", "windows_processed": int(pipe.n_windows), "calibration_windows": int((df["phase"] == "CALIBRATING").sum()),
            "operating_windows": int(len(op)), "decisions": counts, "security_events": len(rec.events), "events_by_kind": ev_kinds,
            "phase": pipe.phase.value, "baseline_version": pipe.registry.current,
            "calibration": calib_note, "sensor_states": {s: st.value for s, st in pipe.security.state.items()},
            "muted_sensors": list(pipe.security.muted_sensors()), "hardware_faults_active": pipe.watchdog.active_faults(),
            "latest_score": _f(last["score"]) if last is not None else None,
            "latest_decision": last["decision"] if last is not None else None,
            "final_drift": _f(last["drift"]) if last is not None else None,
            "ledger_records": len(rec.ledger), "ledger_chain_ok": ok, "ledger_first_bad_seq": bad,
            "input": src.stats.summary(), "evaluation": evaluate(df, rec.manifest), "latency": detection_latency(df, rec.manifest),
        })
    except Exception as e:                                  # noqa: BLE001 - surfaced to the UI, never swallowed silently
        rec.status, rec.error = "failed", f"{type(e).__name__}: {e}"
    finally:
        rec.finished = time.time()
        rec.summary = {**rec.summary, "wall_seconds": round(rec.finished - (rec.started or rec.finished), 1)}


# ---------------------------------------------------------------------------------------------------------------
# persistence (JSON + CSV only)
# ---------------------------------------------------------------------------------------------------------------
def _run_dir(run_id: str, base: Path) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run id {run_id!r}")
    return base / run_id


def save_run(rec: RunRecord, base: Path = OUTPUT_DIR) -> Path:
    d = _run_dir(rec.run_id, base)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": rec.run_id, "params": rec.params, "status": rec.status, "created": rec.created, "started": rec.started,
            "finished": rec.finished, "progress": rec.progress, "error": rec.error, "manifest": rec.manifest,
            "summary": rec.summary, "config": rec.config}
    (d / "run.json").write_text(json.dumps(json_safe(meta), indent=2), encoding="utf-8")
    rec.windows.to_csv(d / "windows.csv", index=False)
    rec.events_df.to_csv(d / "events.csv", index=False)
    (d / "ledger.json").write_text(json.dumps(rec.ledger), encoding="utf-8")
    return d


def load_run(run_dir: Path) -> RunRecord:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rid = meta["run_id"]
    if not RUN_ID_RE.match(rid):
        raise ValueError("bad run id in run.json")
    rec = RunRecord(run_id=rid, params=meta["params"], status=meta.get("status", "completed"), created=meta.get("created", 0.0),
                    started=meta.get("started"), finished=meta.get("finished"), progress=meta.get("progress", 1.0), error=meta.get("error"),
                    manifest=meta.get("manifest", {}), summary=meta.get("summary", {}), config=meta.get("config", {}))
    wf = run_dir / "windows.csv"
    if wf.exists():
        df = pd.read_csv(wf)
        for c in WINDOW_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df["valid"] = df["valid"].astype(bool)
        rec._df, rec._df_n = df[WINDOW_COLUMNS], 0      # loaded runs keep their frame; rows stays empty so the cache is not rebuilt
    ef = run_dir / "events.csv"
    if ef.exists():
        edf = pd.read_csv(ef)
        edf = edf.astype(object).where(edf.notna(), None)
        rec.events = edf.to_dict("records")
    lf = run_dir / "ledger.json"
    if lf.exists():
        rec.ledger = json.loads(lf.read_text(encoding="utf-8"))
    return rec


# ---------------------------------------------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------------------------------------------
class RunManager:
    """Holds all runs of this server process. At most one run executes at a time."""

    def __init__(self, base: Path = OUTPUT_DIR, executor=execute_run) -> None:
        self.base, self._executor = Path(base), executor
        self._lock = threading.Lock()
        self.runs: dict[str, RunRecord] = {}
        self._active: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._load_saved()

    def _load_saved(self) -> None:
        if not self.base.exists():
            return
        for d in sorted(self.base.iterdir()):
            if d.is_dir() and RUN_ID_RE.match(d.name) and (d / "run.json").exists():
                try:
                    rec = load_run(d)
                    self.runs[rec.run_id] = rec
                except Exception:                           # noqa: BLE001 - a corrupt saved run must not stop the app
                    continue

    @property
    def active(self) -> Optional[RunRecord]:
        with self._lock:
            rec = self.runs.get(self._active) if self._active else None
            return rec if rec is not None and rec.status in ("queued", "running") else None

    def start(self, params: RunParams) -> RunRecord:
        with self._lock:
            cur = self.runs.get(self._active) if self._active else None
            if cur is not None and cur.status in ("queued", "running"):
                raise RunBusyError(f"run {cur.run_id} is still in progress; wait for it or cancel it")
            rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            rec = RunRecord(run_id=rid, params=dataclasses.asdict(params))
            self.runs[rid], self._active = rec, rid
            self._thread = threading.Thread(target=self._run_and_save, args=(rec,), name=f"iotguard-run-{rid}", daemon=True)
            self._thread.start()
            return rec

    def _run_and_save(self, rec: RunRecord) -> None:
        self._executor(rec)
        try:
            save_run(rec, self.base)
        except OSError as e:
            rec.error = (rec.error + " | " if rec.error else "") + f"could not save results: {e}"

    def cancel_active(self) -> bool:
        rec = self.active
        if rec is None:
            return False
        rec.cancel.set()
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def ordered(self) -> list[RunRecord]:
        return sorted(self.runs.values(), key=lambda r: r.created, reverse=True)

    def delete(self, run_id: str) -> None:
        import shutil
        rec = self.runs.get(run_id)
        if rec is None or rec.status in ("queued", "running"):
            raise ValueError("run not found or still running")
        d = _run_dir(run_id, self.base)
        if d.exists():
            shutil.rmtree(d)
        del self.runs[run_id]


# ---------------------------------------------------------------------------------------------------------------
# comparison + export
# ---------------------------------------------------------------------------------------------------------------
def _m(ev: dict, key: str, metric: str) -> Optional[float]:
    return ev.get(key, {}).get(metric) if ev.get("available") else None


def comparison_table(records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        s, p = r.summary, r.params
        ev, lat, dec = s.get("evaluation", {}), s.get("latency", {}), s.get("decisions", {})
        rows.append({"run_id": r.run_id, "scenario": p.get("scenario"), "seed": p.get("seed"), "profile": p.get("profile"),
                     "days": (r.manifest.get("duration_s") or 0) / 86400 if r.manifest else p.get("days"), "strength": p.get("strength"),
                     "status": r.status, "partial": s.get("partial"), "windows": s.get("windows_processed"),
                     "operating_windows": s.get("operating_windows"), "TRUSTED": dec.get("TRUSTED"), "UNCERTAIN": dec.get("UNCERTAIN"),
                     "SUSPICIOUS": dec.get("SUSPICIOUS"), "security_events": s.get("security_events"),
                     "detected_first_attack": lat.get("detected") if lat.get("available") else None,
                     "latency_s": lat.get("latency_s") if lat.get("available") else None,
                     "precision": _m(ev, "strict", "precision"), "recall": _m(ev, "strict", "recall"), "f1": _m(ev, "strict", "f1"),
                     "fpr": _m(ev, "strict", "fpr"), "ledger_chain_ok": s.get("ledger_chain_ok"), "wall_seconds": s.get("wall_seconds"),
                     "error": r.error})
    return pd.DataFrame(rows)


def export_json(records: list[RunRecord], include_windows: bool = False) -> str:
    out = []
    for r in records:
        item = {"run_id": r.run_id, "params": r.params, "status": r.status, "error": r.error, "summary": r.summary,
                "manifest": r.manifest, "config": r.config, "events": r.events}
        if include_windows:
            item["windows"] = json_safe(r.windows.to_dict("records"))
        out.append(item)
    return json.dumps(json_safe(out), indent=2)
