"""Engine sessions that run in a background thread so the pages can show them live.

  EngineRun.start_replay(...)  a recording replayed through the real pipeline, with a PIR attack injected in software. The
                               replay pauses when the PIR is quarantined and waits for an admin.
  EngineRun.start_live(...)    the ESP32 on a serial port. The baseline is first learned from a recording (about 10 s), then
                               real readings follow. A software PIR attack can be switched on and off.

Nothing here touches Streamlit: the pages only read snapshots and call admin_action()."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from iotguard.contract import Reading
from iotguard.inject import Injection
from iotguard.pipeline import Phase, Pipeline
from iotguard.presets import demo_setup, room_setup
from iotguard.security import AdminAuth, Ledger, SensorState
from iotguard.sources import ReplaySource

MODE = {"TRUSTED": "full", "UNCERTAIN": "slow", "SUSPICIOUS": "freeze"}
SESSIONS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "sessions"
LIVE_LOGS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "live_logs"      # kept apart from the project's logs folder
WINDOW_S = 30.0
DRAIN_H = 1.5


class EngineRun:
    def __init__(self, kind: str, title: str, auth: AdminAuth, boot_secret: str, demo: float = 2.0, sessions_dir: Path = SESSIONS_DIR) -> None:
        self.kind, self.title, self.demo = kind, title, demo
        self._boot_secret = boot_secret
        self.lock = threading.RLock()
        self.stop_ev, self.resume_ev = threading.Event(), threading.Event()
        self.status, self.note, self.error = "starting", "", None
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.marks: list[dict] = []
        self.raw: deque = deque(maxlen=1500)
        self.latest: Optional[dict] = None
        self.attack_on = kind == "replay"            # replay: attack runs until an admin deals with it. live: a switch on the page
        self.warm = False
        self.t0_h = 0.0                               # live: device hour where real readings begin
        self.src_stats = None
        self.log_path: Optional[Path] = None
        self.log_lines = 0
        self.threshold: Optional[float] = None
        self.cal_end_h: Optional[float] = None
        self.atk_span: Optional[tuple[float, float]] = None
        self._attacking = False
        self._pause_flag = False
        self._deferred = None
        self.speed = 180.0                            # simulated minutes per real second while paced
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = sessions_dir / f"{kind}_{stamp}_ledger.jsonl"
        self.ledger = Ledger(self.ledger_path)
        auth.ledger = self.ledger                     # wrong passwords and lockouts land in this session's ledger
        self.auth = auth
        cfg, cons = demo_setup(demo) if demo else room_setup()
        self.cfg = cfg
        self.threshold = cfg.anomaly.mahalanobis_threshold
        self.pipe = Pipeline(cfg, consensus=cons, ledger=self.ledger, auth=auth, naive_twin=True)
        self.thread: Optional[threading.Thread] = None

    # ---- reading the state (UI thread) --------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def sensor_states(self) -> dict[str, str]:
        with self.lock:
            return {s: v.value for s, v in self.pipe.security.state.items()}

    def quarantined(self) -> list[str]:
        return [s for s, v in self.sensor_states().items() if v == SensorState.QUARANTINED.value]

    def frame(self) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(self.rows)

    def raw_frame(self) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(list(self.raw))

    def ledger_records(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.ledger.records]

    def stats_text(self) -> str:
        s = self.src_stats
        try:
            return s.summary() if s is not None else ""
        except Exception:
            return ""

    # ---- control -------------------------------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_ev.set()
        self.resume_ev.set()

    def admin_action(self, sensor: str, action: str, credential: str, note: str = "") -> str:
        """Release a quarantined sensor (action 'unquarantine') or record a replacement then release it ('replace').
        The pipeline checks the credential itself and raises AuthError; nothing is written to the ledger before it passes."""
        if action not in ("unquarantine", "replace"):
            raise ValueError(action)
        with self.lock:
            t = self.pipe._last_t or 0.0
            self.pipe.auth.require(credential, t)
            if action == "replace":
                self.ledger.append("sensor_replaced", t, sensor=sensor,
                                   detail=note or "faulty sensor swapped for a new unit; the new unit starts on probation")
            if self.kind == "replay":
                # The detector's motion counters still hold the attack. Releasing now would fail probation at once, so the
                # approval is recorded now and the release is applied after DRAIN_H hours of clean data (the validated delay).
                self.ledger.append("admin_release_approved", t, sensor=sensor, action=action, detail=f"release applies after {DRAIN_H:g} h of clean data")
                self._deferred = (sensor, credential, t + DRAIN_H * 3600.0)
                self.attack_on = False               # the fault is gone: a new sensor, or the old one inspected and cleared
                msg = f"approved; the sensor stays masked for {DRAIN_H:g} h of clean data, then probation starts"
                self.note = msg
            else:
                msg = self.pipe.release_sensor(sensor, credential, t).detail
                self.note = f"{sensor.upper()} on probation"
            self.marks.append({"kind": "sensor_replaced" if action == "replace" else "admin_release", "t_h": t / 3600.0, "detail": sensor})
            self._pause_flag = False
            self.status = "running"
        self.resume_ev.set()
        return msg

    # ---- the shared feed -------------------------------------------------------------------------------------
    def _feed(self, r: Reading) -> None:
        with self.lock:
            outs = self.pipe.push(r)
            for o in outs:
                self._record(o)

    def _record(self, o) -> None:
        t_h = o.t_start / 3600.0
        if o.calibration is not None:
            self.cal_end_h = t_h
            self.marks.append({"kind": "calibration_done", "t_h": t_h, "detail": o.calibration.summary()})
        if o.decision is None:
            return
        for e in o.events:
            if not self.warm:
                self.events.append({"kind": e.kind, "t_h": round(e.t / 3600.0, 3), "sensor": e.sensor or "", "detail": e.detail})
            if e.kind == "sensor_quarantined" and self.kind == "replay" and not self.warm:
                self._pause_flag = True
        if self.warm:
            return
        if self.kind == "live" and t_h < self.t0_h - 1e-9:
            return                                               # the window that was still open when the baseline finished
        d = o.decision
        self.rows.append({"t_h": t_h, "label": d.label.value, "mode": MODE[d.label.value], "alpha": o.alpha, "score": o.score,
                          "drift": o.drift, "naive_drift": o.naive_drift, "occupancy": d.occupancy.value if d.occupancy is not None else "",
                          "muted": ",".join(o.muted)})
        if len(self.rows) > 60000:
            del self.rows[:10000]

    def _wait_admin(self) -> None:
        with self.lock:
            q = self.quarantined()
            self.status = "waiting_admin"
            self.note = ", ".join(s.upper() for s in q) + " quarantined: the replay is paused until an admin deals with it"
            self.resume_ev.clear()
        while not self.resume_ev.wait(0.25):
            if self.stop_ev.is_set():
                break

    # ---- replay ----------------------------------------------------------------------------------------------
    def start_replay(self, path: str, attack_start_h: float, attack_hours: float, speed: float = 180.0, lead_h: float = 1.0) -> None:
        self.speed = speed
        self.thread = threading.Thread(target=self._replay_main, args=(path, attack_start_h, attack_hours, lead_h), daemon=True, name="replay")
        self.thread.start()

    def _replay_main(self, path: str, a_start: float, a_hours: float, lead_h: float) -> None:
        src = None
        try:
            src = ReplaySource(path)
            inj = Injection("pir", a_start, a_hours, "force", 1, tag="pir_stuck_high")
            self.atk_plan = (a_start, a_start + a_hours)
            self.pipe.start_calibration(self._boot_secret)
            self.status, self.note = "calibrating", "learning the baseline from the first part of the recording (fast-forward)"
            t0, last_t, owed, recovered = None, None, 0.0, False
            for r in src:
                if self.stop_ev.is_set():
                    break
                if t0 is None:
                    t0 = r.t
                h = (r.t - t0) / 3600.0
                on = self.attack_on and inj.active(h)
                if on and not self._attacking:
                    self._attacking = True
                    self.atk_span = (r.t / 3600.0, None)
                    self.marks.append({"kind": "attack_start", "t_h": r.t / 3600.0, "detail": "PIR forced high (software only)"})
                elif self._attacking and not on:
                    self._attacking = False
                    self.atk_span = (self.atk_span[0], r.t / 3600.0)
                    self.marks.append({"kind": "attack_end", "t_h": r.t / 3600.0, "detail": "the PIR reports real data again"})
                self._feed(r if not on else inj.apply(r))
                if self._deferred is not None and r.t >= self._deferred[2]:
                    sensor, cred, _ = self._deferred
                    self._deferred = None
                    with self.lock:
                        try:
                            self.pipe.release_sensor(sensor, cred, r.t)
                            self.note = f"{sensor.upper()} on probation: {self.cfg.security.probation_windows} clean windows to recover"
                        except Exception as e:
                            self.note = f"release failed: {e}"
                if self._pause_flag:
                    self._wait_admin()
                    if self.stop_ev.is_set():
                        break
                if self.pipe.phase is Phase.OPERATING:
                    if self.status == "calibrating":
                        self.status, self.note = "running", ""
                    if any(e["kind"] == "sensor_recovered" for e in self.events[-5:]):
                        recovered = True
                    if h >= a_start - lead_h and not recovered:           # slow down near the interesting part, so it can be watched
                        if last_t is not None:
                            owed += (r.t - last_t) / (self.speed * 60.0)
                            if owed >= 0.04:
                                time.sleep(owed)
                                owed = 0.0
                last_t = r.t
            self.status = "stopped" if self.stop_ev.is_set() else "finished"
            if self.pipe.phase is Phase.CALIBRATION_FAILED or (self.pipe.phase is Phase.CALIBRATING and self.status == "finished"):
                self.status, self.error = "error", "calibration did not finish: the recording is too short for this calibration length"
        except Exception as e:                                  # shown on the page; the server keeps running
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
        finally:
            if src is not None:
                src.close()

    # ---- live ------------------------------------------------------------------------------------------------
    def start_live(self, warm_path: str, source_factory: Callable[[], Iterable[Reading]]) -> None:
        self.thread = threading.Thread(target=self._live_main, args=(warm_path, source_factory), daemon=True, name="live")
        self.thread.start()

    def _live_main(self, warm_path: str, source_factory: Callable[[], Iterable[Reading]]) -> None:
        src = None
        logf = None
        try:
            self.status, self.note, self.warm = "calibrating", f"learning the baseline from {Path(warm_path).name} (about 10 seconds)", True
            self.pipe.start_calibration(self._boot_secret)
            warm_src = ReplaySource(warm_path)
            for r in warm_src:
                if self.stop_ev.is_set():
                    return
                self._feed(r)
            warm_src.close()
            if self.pipe.phase is not Phase.OPERATING:
                raise RuntimeError("the baseline could not be learned from that recording (too short for a calibration of this length)")
            with self.lock:
                self.pipe.engine.reset_occupancy()                      # who was in the room when the recording ended is stale
                self.pipe.gate.reset()
                last = self.pipe._last_t or 0.0
                base = (int(last // WINDOW_S) + 3) * WINDOW_S           # real readings continue on the same time line
                self.t0_h = base / 3600.0
                self.warm = False
                self.rows.clear()
                self.events.clear()
            inj = Injection("pir", 0.0, 1e9, "force", 1, tag="live_pir_attack")
            self.status, self.note = "running", "live"
            src = source_factory()
            self.src_stats = getattr(src, "stats", None)
            LIVE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LIVE_LOGS_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            logf = open(self.log_path, "x", encoding="utf-8")                  # 'x': never overwrites an existing file
            logf.write(f"# live readings from the dashboard, started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            for r in src:
                if self.stop_ev.is_set():
                    break
                logf.write(json.dumps({"t": r.t, "temp": r.temp, "hum": r.hum, "pir": r.pir, "door": r.door, "ldr": r.ldr}, separators=(",", ":")) + "\n")
                self.log_lines += 1
                if self.log_lines % 20 == 0:
                    logf.flush()                                              # the real values, before any software attack
                r = replace(r, t=base + r.t)
                shown = r
                on = self.attack_on
                if on:
                    r = inj.apply(r)
                if on and not self._attacking:
                    self._attacking = True
                    self.marks.append({"kind": "attack_start", "t_h": r.t / 3600.0, "detail": "PIR forced high (software only)"})
                elif self._attacking and not on:
                    self._attacking = False
                    self.marks.append({"kind": "attack_end", "t_h": r.t / 3600.0, "detail": "software attack off"})
                with self.lock:
                    self.latest = {"t_h": r.t / 3600.0, "temp": shown.temp, "hum": shown.hum, "pir": shown.pir, "door": shown.door, "ldr": shown.ldr,
                                   "pir_used": r.pir}
                    self.raw.append(dict(self.latest))
                self._feed(r)
            self.status = "stopped"
        except Exception as e:
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
        finally:
            self.warm = False
            if logf is not None:
                logf.close()
            if src is not None:
                close = getattr(src, "close", None)
                if close:
                    close()


# ---- helpers for the pages ----------------------------------------------------------------------------------------
def find_recordings() -> list[Path]:
    """Recordings (.jsonl) in ../logs, ./data and $IOTGUARD_LOGS, merged ones first."""
    import os
    root = Path(__file__).resolve().parents[1]
    dirs = [root.parent / "logs", root / "data", root / "outputs" / "recordings"]
    if os.environ.get("IOTGUARD_LOGS"):
        dirs.append(Path(os.environ["IOTGUARD_LOGS"]))
    found = [p for d in dirs if d.is_dir() for p in d.glob("*.jsonl") if "ledger" not in p.name and not p.name.startswith("pipeline_")]
    return sorted(found, key=lambda p: ("merged" not in p.name.lower(), -p.stat().st_size))


def list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []
