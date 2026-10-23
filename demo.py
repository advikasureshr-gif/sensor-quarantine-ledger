"""
demo.py - run the project's demo acts with clean, readable output.

    python demo.py all                    legit change, then slow poisoning, then PIR quarantine (about 4 minutes)
    python demo.py quarantine             one act:  legit | poison | quarantine | mixed
    python demo.py all --pause            wait for Enter between acts, so you can talk
    python demo.py poison --plot          also save the drift figure (adds about a minute) to figs/
    python demo.py hardware --port COM3   live ESP32 stream (Ctrl+C to stop; recording saved to logs/)
    python demo.py list

Every act is a SIMULATION run through the real pipeline (the simulator's labels are never shown to the system).
The screen shows only events and daily reports; the full per-window output is saved to logs/demo_<act>_<time>.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS, FIGS = HERE / "logs", HERE / "figs"

# key: (title, scenario, seed, days or None, what to say, what to watch for)
ACTS = {
    "legit": ("A real change is learned, not blocked", "legit_warmer", 0, 6,
              "The room starts running 4 C warmer on day 3 (a season or AC change). That is NOT an attack.",
              "No alerts, no quarantine. Both baselines move together: the defence must not stop genuine change."),
    "poison": ("Slow poisoning: the gated baseline resists", "gradual_poison", 1, 7,
               "From day 3 an attacker slowly bends several sensors, a little each hour, to teach the system a false 'normal'.",
               "The [day report] lines: the always-learning twin drifts away from the calibrated baseline, the gated one does not. "
               "The PIR is quarantined."),
    "quarantine": ("PIR stuck high at night: alert, then quarantine", "pir_stuck_night", 0, 5,
                   "At night the room is empty, but the PIR reads 'motion' for 4 hours with no door opening to explain it.",
                   "ALERT (motion with no entry), then SENSOR_QUARANTINED [pir] after 3 h of suspicion. The other sensors keep "
                   "working, and the ledger check at the end says 'intact'."),
    "mixed": ("Real change first, then attacks", "mixed_legit_attack", 0, None,
              "The room gets brighter (learned), then two attacks follow.",
              "The change is adopted, the attacks are caught: the system is not over-conservative."),
}
ORDER = ["legit", "poison", "quarantine"]
SHOW = ("***", "[day report]", "done", "  sensor ", "  hardware", "  input", "DEMO")


def banner(text: str, ch: str = "=") -> None:
    print("\n" + ch * 78 + "\n" + text + "\n" + ch * 78, flush=True)


def run_act(key: str, a: argparse.Namespace) -> int:
    title, scenario, seed, days, say, watch = ACTS[key]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    LOGS.mkdir(exist_ok=True)
    log_path, ledger = LOGS / f"demo_{key}_{stamp}.log", LOGS / f"demo_{key}_{stamp}_ledger.jsonl"
    banner(f"ACT: {title}\n  scenario: {scenario}   seed: {seed}")
    print(f"WHAT IS HAPPENING: {say}\nWHAT TO WATCH FOR: {watch}\n", flush=True)
    cmd = [sys.executable, "-u", "-m", "iotguard.run", "--simulate", scenario, "--seed", str(seed), "--naive", "--quiet",
           "--ledger", str(ledger)] + (["--days", str(days)] if days else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    print("(simulating several days of sensor data; this takes about a minute)\n", flush=True)
    t0, hidden = time.time(), 0
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", env=env)
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            if line.startswith(SHOW):
                print(line.rstrip(), flush=True)
            else:
                hidden += 1
    code = proc.wait()
    print(f"\n[{time.time() - t0:.0f} s; {hidden} per-window lines are in {log_path.relative_to(HERE)}]", flush=True)
    if code != 0:
        print(f"!! the run exited with code {code}; see the log", flush=True)
        return code
    if a.plot:
        out = FIGS / f"demo_{key}.png"
        print(f"drawing the figure -> {out.relative_to(HERE)} ...", flush=True)
        pc = [sys.executable, "plot_drift.py", scenario, "--seed", str(seed), "--out", str(out)] + (["--days", str(days)] if days else [])
        code = subprocess.call(pc, cwd=HERE, env=env, stdin=subprocess.DEVNULL)
    return code


def run_hardware(a: argparse.Namespace) -> int:
    if not a.port:
        print("hardware needs --port, e.g.  python demo.py hardware --port COM3\n"
              "(find it with  python -m iotguard.record --list ; close the Arduino Serial Monitor first)")
        return 2
    LOGS.mkdir(exist_ok=True)
    out = LOGS / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    banner("LIVE HARDWARE: the ESP32 streams into the same pipeline input\n"
           "  wave at the PIR, move the magnet at the reed switch, cover the LDR. Ctrl+C to stop.")
    cmd = [sys.executable, "-u", "-m", "iotguard.record", "--port", a.port, "--out", str(out)]
    try:
        return subprocess.call(cmd, cwd=HERE)
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acts", nargs="+", help="all | " + " | ".join(ACTS) + " | hardware | list")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between acts")
    ap.add_argument("--plot", action="store_true", help="also save the drift figure for each act")
    ap.add_argument("--port", help="serial port for the hardware act, e.g. COM3")
    a = ap.parse_args(argv)

    if a.acts == ["list"]:
        for k, (title, sc, seed, days, _, _) in ACTS.items():
            print(f"{k:<11} {title}   [{sc}, seed {seed}]")
        print(f"{'hardware':<11} live ESP32 stream (needs --port)")
        return 0
    todo: list[str] = []
    for k in a.acts:
        todo += ORDER if k == "all" else [k]
    bad = [k for k in todo if k not in ACTS and k != "hardware"]
    if bad:
        ap.error(f"unknown act(s): {', '.join(bad)} (try: python demo.py list)")
    rc = 0
    for i, k in enumerate(todo):
        rc = run_hardware(a) if k == "hardware" else run_act(k, a)
        if rc != 0:
            return rc
        if a.pause and i < len(todo) - 1:
            input("\nPress Enter for the next act...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
