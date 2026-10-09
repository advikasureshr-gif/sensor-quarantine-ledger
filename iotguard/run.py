"""
run.py - run the whole system on live data, a recorded log, or a simulated scenario.

    python -m iotguard.run --port COM3                       live ESP32 (calibrates first, then operates)
    python -m iotguard.run --replay logs/s1.jsonl            a recording made with iotguard.record
    python -m iotguard.run --simulate pir_stuck_night        a simulator scenario (labels are never shown to the system)

Options:  --demo FACTOR   shorten the long timers (calibration etc.) by FACTOR for a tabletop demo (UNVALIDATED, see presets.py)
          --ledger FILE   write the tamper-evident ledger there      --quiet   print only non-TRUSTED windows and events
          --state FILE    save the whole system state there (autosaved, and on exit) and RESUME from it if it exists, so a
                          restart does not cost a calibration or clear a quarantine. Use the same --ledger so edits are detected.
          --naive         also run an always-learning twin and print the daily baseline drift of both (the project's key comparison)
          --admin-env VAR read the admin secret from environment variable VAR (enables authentication)
While running, type "release <sensor>" (dht, pir, door or ldr) to start probation for a quarantined sensor, or
"recalibrate", using the admin secret from --admin-env.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from typing import Callable, Iterable, Optional

from pathlib import Path

from .config import Config
from .consensus import Trust
from .pipeline import Phase, Pipeline, StateIntegrityError, StateMismatchError, WindowOutcome
from .presets import demo_setup, room_setup
from .security import AdminAuth, AuthError, Ledger
from .sources import ReplaySource, SerialSource, Source, strip_labels


def _main(argv: Optional[list[str]] = None, *, source: Optional[Iterable] = None, input_lines: Optional[Iterable[str]] = None,
          echo: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="python -m iotguard.run", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=source is None)
    g.add_argument("--port"); g.add_argument("--replay"); g.add_argument("--simulate")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=None, help="length of a simulated scenario")
    ap.add_argument("--speed", type=float, default=None, help="replay pacing (1 = real time); default: as fast as possible")
    ap.add_argument("--demo", type=float, default=None, metavar="FACTOR")
    ap.add_argument("--ledger"); ap.add_argument("--quiet", action="store_true"); ap.add_argument("--admin-env"); ap.add_argument("--state"); ap.add_argument("--naive", action="store_true")
    ap.add_argument("--log", help="also write everything printed to this text file (each line stamped with the wall-clock time)")
    a = ap.parse_args(argv)

    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        logf = open(a.log, "a", encoding="utf-8", buffering=1)
        plain_echo = echo
        def echo(msg: str = "") -> None:                       # noqa: F811 - same name on purpose: every message goes to both
            plain_echo(msg)
            for line in str(msg).splitlines() or [""]:
                logf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    else:
        logf = None
    old_term = None
    try:
        old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # kill / service stop = Ctrl+C
    except (ValueError, OSError):
        pass                                                   # not in the main thread (tests) or unsupported

    cfg, cons = demo_setup(a.demo) if a.demo else room_setup()
    ledger = Ledger(a.ledger) if a.ledger else Ledger()
    auth, secret = None, None
    if a.admin_env:
        secret = os.environ.get(a.admin_env)
        if not secret:
            ap.error(f"environment variable {a.admin_env} is empty or not set")
        auth = AdminAuth(secret, ledger=ledger)
    else:
        echo("note: no --admin-env given, so admin actions are not authenticated (fine for simulation and testing)")
    if a.demo:
        echo(f"DEMO preset x{a.demo:g}: long timers shortened; this preset has not been validated (see presets.py)")
    resumed = False
    if a.state and Path(a.state).exists():
        try:
            pipe = Pipeline.load(a.state, cfg, cons, ledger=ledger, auth=auth, state_path=a.state)
        except (StateIntegrityError, StateMismatchError, ValueError) as e:
            echo(f"cannot resume from {a.state}: {e}")
            return 2
        resumed = True
        echo(f"resumed from {a.state}: phase {pipe.phase.value}, {pipe.n_windows} windows already processed, baseline version {pipe.registry.current}")
        for sname, st in pipe.security.state.items():
            if st.value != "ACTIVE":
                echo(f"  sensor {sname} is still {st.value}")
    else:
        pipe = Pipeline(cfg, consensus=cons, ledger=ledger, auth=auth, state_path=a.state, naive_twin=a.naive)

    if source is None:
        if a.port:
            source = SerialSource(a.port, a.baud, on_event=lambda m: echo(f"[serial] {m}"))
        elif a.replay:
            source = ReplaySource(a.replay, realtime=a.speed)
        else:
            from .scenarios import make_source
            source = strip_labels(make_source(a.simulate, a.seed, days=a.days))

    def commands() -> None:
        try:
            for line in (input_lines if input_lines is not None else sys.stdin):
                w = line.split()
                try:
                    if len(w) == 2 and w[0] == "release":
                        echo("  " + pipe.release_sensor(w[1], secret, pipe.n_windows * cfg.features.window_s).detail)
                    elif w == ["recalibrate"]:
                        pipe.start_calibration(secret, pipe.n_windows * cfg.features.window_s); echo("  recalibration started")
                    elif w:
                        echo("  commands: release <dht|pir|door|ldr>, recalibrate")
                except (AuthError, ValueError) as e:
                    echo(f"  refused: {e}")
        except Exception:
            pass
    threading.Thread(target=commands, daemon=True).start()

    if not resumed or pipe.phase is Phase.IDLE:
        pipe.start_calibration(secret)
        echo(f"calibrating: needs {cfg.calibration.min_duration_s / 3600:.1f} h of data and {cfg.calibration.min_trusted_windows} trusted windows"
             + (" (Ctrl+C to stop)" if a.port else ""))
    elif pipe.phase is Phase.CALIBRATION_FAILED:
        echo("the previous calibration failed; type 'recalibrate' (admin) to try again")
    shown_cal = 0
    next_report = [None]

    def show(o: WindowOutcome) -> None:
        nonlocal shown_cal
        for e in o.events:
            echo(f"*** {e.kind.upper()} at t={e.t:.0f}s" + (f" [{e.sensor}]" if e.sensor else "") + f": {e.detail}")
        if o.phase is Phase.CALIBRATING and o.calibration is None:
            shown_cal += 1
            if shown_cal % 120 == 0 and pipe.calibrator is not None:
                echo(f"  calibrating... {pipe.calibrator.span_s / 3600:.1f} h collected, {pipe.calibrator.n_candidates} windows")
        elif o.calibration is not None:
            echo(o.line())
        elif o.phase is Phase.OPERATING and o.decision is not None and not (a.quiet and o.decision.label is Trust.TRUSTED):
            echo(o.line())
        if a.naive and o.phase is Phase.OPERATING and o.decision is not None and not math.isnan(o.divergence):
            if next_report[0] is None:
                next_report[0] = o.t_start + 86400.0
            if o.t_start >= next_report[0]:
                next_report[0] += 86400.0
                echo(f"[day report] t={o.t_start / 86400:.1f} d  baseline drift: gated {o.drift:.2f}   always-learning twin {o.naive_drift:.2f}   "
                     f"distance between them {o.divergence:.2f}")

    try:
        pipe.run(source, on_outcome=show)
    except KeyboardInterrupt:
        pass
    except (OSError, ImportError) as e:
        echo(f"stopped: {e}")
    finally:
        if isinstance(source, Source):
            source.close()
    ok, bad = ledger.verify()
    echo(f"\ndone: {pipe.n_windows} windows, phase {pipe.phase.value}, baseline version {pipe.registry.current}, "
         f"{len(pipe.security.events)} security event(s), ledger {'intact' if ok else f'BROKEN at {bad}'}")
    for s, st in pipe.security.state.items():
        if st.value != "ACTIVE":
            echo(f"  sensor {s}: {st.value}")
    for fault, since in pipe.watchdog.active_faults().items():
        echo(f"  hardware fault still active: {fault} (since t={since:.0f}s)")
    if isinstance(source, Source):
        echo(f"  input: {source.stats.summary()}")
    if a.state and pipe.state_path is not None:
        echo(f"  state saved to {a.state}")
    if old_term is not None:
        signal.signal(signal.SIGTERM, old_term)
    if logf is not None:
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main())