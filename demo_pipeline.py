"""
demo_pipeline.py - the WHOLE pipeline on one recording, with one sensor attacked in software and quarantined.

    python demo_pipeline.py --replay logs/room.jsonl --attack-start-h 56
    python demo_pipeline.py --replay logs/room.jsonl --attack-start-h 14 --demo 4        (shorter calibration, unvalidated)
    python demo_pipeline.py --make-sim logs/sim6d.jsonl                                  (a simulated stand-in recording)
    python -m iotguard.inject --suggest logs/room.jsonl --min-start-h 50                 (find a good attack time)

What it shows, in order:
  1 input         the recording (real ESP32 readings) replayed as a live stream
  2 calibration   admin-authenticated; the baseline is learned from trusted windows only and stored as version 1
  3 operation     normal windows are TRUSTED and the baseline keeps learning
  4 attack        the PIR is forced high for hours IN SOFTWARE (no sensor is touched), while the room is empty
  5 detection     motion with no door opening: consensus says SUSPICIOUS, learning is frozen, an ALERT is raised
  6 quarantine    after hours of sustained suspicion the PIR alone is quarantined; the other sensors keep working
  7 recovery      a wrong admin secret is refused; the right one starts probation; clean data restores the sensor
  8 provenance    every step is in a hash-chained ledger; editing one record is detected
"""
from __future__ import annotations

import argparse
import collections
import re
import shutil
import sys
import time
from pathlib import Path

from iotguard.consensus import Trust
from iotguard.inject import Injection, InjectSource
from iotguard.pipeline import Phase, Pipeline
from iotguard.presets import demo_setup, room_setup
from iotguard.runexport import RunRecorder
from iotguard.security import AdminAuth, AuthError, Ledger, SensorState
from iotguard.sources import ReplaySource

HERE = Path(__file__).resolve().parent


def say(text: str = "") -> None:
    print(text, flush=True)


def banner(n: str, title: str) -> None:
    say("\n" + "=" * 78 + f"\n[{n}] {title}\n" + "=" * 78)


def make_sim(path: str, days: float, seed: int) -> int:
    from iotguard.scenarios import make_source
    from iotguard.sources import strip_labels, write_jsonl
    n = write_jsonl(strip_labels(make_source("normal", seed, days=days)), path, include_labels=False)
    say(f"wrote {n} simulated readings ({days:g} days) to {path}\n"
        f"try:  python demo_pipeline.py --replay {path} --attack-start-h 86")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="recording made with iotguard.record (.jsonl)")
    ap.add_argument("--attack-start-h", type=float, help="when the PIR attack starts, in hours since the recording began")
    ap.add_argument("--attack-hours", type=float, default=4.5, help="attack length; quarantine needs at least ~3.2 h (default 4.5)")
    ap.add_argument("--release-after-h", type=float, default=1.5,
                    help="hours after the attack ends before the admin releases the sensor (its motion counters need time to drain)")
    ap.add_argument("--demo", type=float, metavar="FACTOR", help="shorten calibration by this factor (unvalidated preset)")
    ap.add_argument("--secret", default="demo-admin-secret", help="admin secret used for calibration and release (min 8 characters)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter at each stage so you can talk")
    ap.add_argument("--export-dir", metavar="DIR", help="save this run as files for the dashboard (default: exports/<timestamp>)")
    ap.add_argument("--no-export", action="store_true", help="do not save the run's files")
    ap.add_argument("--make-sim", metavar="FILE", help="write a simulated recording to FILE and exit")
    ap.add_argument("--sim-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.make_sim:
        return make_sim(a.make_sim, a.sim_days, a.seed)
    if not a.replay or a.attack_start_h is None:
        ap.error("--replay FILE and --attack-start-h HOURS are required (find a time with: python -m iotguard.inject --suggest FILE)")
    if not Path(a.replay).exists():
        ap.error(f"{a.replay} does not exist")

    def wait(msg: str = "press Enter to continue") -> None:
        if a.pause:
            input(f"   ({msg}) ")

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ledger_path = logs / f"pipeline_{stamp}_ledger.jsonl"
    ledger = Ledger(ledger_path)
    auth = AdminAuth(a.secret, ledger=ledger)
    cfg, cons = demo_setup(a.demo) if a.demo else room_setup()
    pipe = Pipeline(cfg, consensus=cons, ledger=ledger, auth=auth, naive_twin=True)

    # ---- 1 input -----------------------------------------------------------------------------------------------------
    banner("1", "INPUT: a recording of the real sensors, replayed as a live stream")
    replay = ReplaySource(a.replay)
    inj = Injection("pir", a.attack_start_h, a.attack_hours, "force", 1, tag="pir_stuck_high")
    ctx = {"atk_start": None, "atk_end": None, "released": False, "recovered": False, "cal_done": False, "last_hr": -1}
    rec = RunRecorder()
    counts = {p: collections.Counter() for p in ("before the attack", "during the attack", "after the attack")}

    def period(t: float) -> str:
        if ctx["atk_start"] is None:
            return "before the attack"
        return "during the attack" if ctx["atk_end"] is None else "after the attack"

    def on_change(ev: str, i: Injection, t: float) -> None:
        if ev == "start":
            ctx["atk_start"] = t
            rec.mark("attack_start", t, f"{i.channel} forced to {int(i.value)} for {i.hours:g} h (software only)")
            banner("4", f"ATTACK: the PIR is forced HIGH for {i.hours:g} h (device hour {t / 3600:.1f}); no sensor is touched")
            say("   The room is empty and no door has opened, so motion cannot be real: it is 'motion with no entry'.")
            wait("attack starting")
        else:
            ctx["atk_end"] = t
            rec.mark("attack_end", t, "the sensor reports real data again")
            say(f"\n   -- attack window over at device hour {t / 3600:.1f}; the PIR reports real data again --")

    source = InjectSource(replay, [inj], on_change=on_change)
    say(f"   file: {a.replay}")
    say(f"   preset: {'DEMO x%g (shortened calibration, unvalidated)' % a.demo if a.demo else 'room (calibration needs 2 days of data)'}")

    # ---- 2 calibration ----------------------------------------------------------------------------------------------
    banner("2", "CALIBRATION: admin-authenticated; learns 'normal' from trusted windows only")
    pipe.start_calibration(a.secret)
    say(f"   calibration started (admin secret checked); needs {cfg.calibration.min_duration_s / 3600:.1f} h of data "
        f"and {cfg.calibration.min_trusted_windows} trusted windows")

    def recover_step(o) -> None:
        t = o.t_start
        banner("7", "ADMIN RECOVERY: the quarantined PIR is released only by an authenticated admin")
        try:
            pipe.release_sensor("pir", "not-the-secret", t)
            say("   !! a wrong secret was accepted (this should not happen)")
        except AuthError as e:
            say(f"   wrong secret -> REFUSED ({e}). The failed attempt is recorded in the ledger.")
        ev = pipe.release_sensor("pir", a.secret, t)
        ctx["released"] = True
        rec.mark("admin_release", t, "wrong secret refused, correct secret accepted")
        say(f"   correct secret -> {ev.detail}")
        say(f"   the PIR must now show {cfg.security.probation_windows} clean windows "
            f"({cfg.security.probation_windows * cfg.features.window_s / 3600:.1f} h) before it influences learning again...")
        wait("released")

    def on_outcome(o) -> None:
        hr = int(o.t_start // 3600)
        if o.calibration is not None:
            ctx["cal_done"] = True
            ctx["cal_t"] = o.t_start
            ctx["cal_text"] = o.calibration.summary()
            rec.mark("calibration_done", o.t_start, o.calibration.summary())
            ok = pipe.phase is Phase.OPERATING
            banner("3", "BASELINE " + ("STORED, NORMAL OPERATION BEGINS" if ok else "FAILED"))
            say("   " + o.calibration.summary())
            say(f"   baseline version {pipe.registry.current} recorded in the ledger (device hour {o.t_start / 3600:.1f})")
            wait("calibrated")
        elif o.phase is Phase.OPERATING and hr != ctx["last_hr"] and hr % 12 == 0 and ctx["atk_start"] is None:
            say(f"   ... normal operation, device hour {hr}")
        ctx["last_hr"] = hr
        if o.phase is Phase.OPERATING and o.decision is not None:
            counts[period(o.t_start)][o.decision.label.value] += 1
            rec.window(o, period(o.t_start))
        for e in o.events:
            rec.event(e)
            if e.kind == "alert":
                banner("5", f"DETECTION at device hour {e.t / 3600:.1f}: ALERT")
                say(f"   {e.detail}")
                say("   SUSPICIOUS windows freeze learning, so the attacker cannot teach the baseline a false 'normal'.")
                wait("alert")
            elif e.kind == "sensor_quarantined":
                banner("6", f"QUARANTINE at device hour {e.t / 3600:.1f}: sensor {e.sensor.upper()}")
                say(f"   {e.detail}")
                say(f"   The {e.sensor} is masked out of scoring and learning. The DHT, door and LDR keep working.")
                wait("quarantined")
            elif e.kind == "sensor_recovered":
                ctx["recovered"] = True
                say(f"\n   RECOVERED at device hour {e.t / 3600:.1f}: sensor {e.sensor.upper()} ({e.detail})")
            elif e.kind == "probation_failed":
                say(f"\n   probation failed at device hour {e.t / 3600:.1f}: {e.detail}")
        if (not ctx["released"] and ctx["atk_end"] is not None and o.t_start >= ctx["atk_end"] + a.release_after_h * 3600.0
                and pipe.security.state["pir"] is SensorState.QUARANTINED):
            recover_step(o)

    try:
        pipe.run(source, on_outcome=on_outcome)
    except KeyboardInterrupt:
        say("\n(stopped)")
    finally:
        source.close()

    st = replay.stats
    say(f"\n   input: {st.summary()}")
    if not ctx["cal_done"]:
        say("\n!! Calibration never finished: the recording is too short or too noisy for it.")
        say("   Record longer, or try --demo 4 (calibration 4x shorter).")
        return 1
    if ctx["atk_start"] is None:
        say("\n!! The recording ended before the attack time. Use a smaller --attack-start-h.")
        return 1

    # ---- 8 provenance -----------------------------------------------------------------------------------------------
    banner("8", "PROVENANCE: every event above is in a hash-chained ledger")
    ok, bad = ledger.verify()
    kinds = collections.Counter(r["event"] for r in ledger.records)
    say(f"   {len(ledger.records)} records in {ledger_path.relative_to(HERE)}: " + ", ".join(f"{k} x{v}" for k, v in kinds.items()))
    say("   last records (each hash covers the previous one):")
    for r in ledger.records[-4:]:
        say(f"     #{r['seq']:<4} {r['event']:<22} prev {r['prev'][:10]}...  hash {r['hash'][:10]}...")
    say(f"   chain check: {'INTACT' if ok else 'BROKEN at record %s' % bad}")
    victim = next((r for r in ledger.records if r["event"] == "sensor_quarantined"), None)
    if victim is not None:
        tampered = ledger_path.with_name(ledger_path.stem + "_tampered.jsonl")
        shutil.copy(ledger_path, tampered)
        lines = tampered.read_text(encoding="utf-8").splitlines()
        lines[victim["seq"]] = lines[victim["seq"]].replace('"sensor":"pir"', '"sensor":"ldr"')
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
        say(f"\n   Tamper test: an attacker edits a copy to blame the LDR instead of the PIR (record #{victim['seq']}):")
        try:
            Ledger(tampered)
            say("   !! the edit was NOT detected")
        except ValueError as e:
            m = re.search(r"first bad record: (\d+)", str(e))
            say(f"   -> DETECTED: the chain breaks at record #{m.group(1) if m else '?'}; the edit cannot be hidden")

    # ---- blockchain -------------------------------------------------------------------------------------------------
    banner("8b", "BLOCKCHAIN: the ledger batched into mined blocks and held by 3 replicas that vote")
    from iotguard.chain import Blockchain, Network
    chain = Blockchain.build([dict(r) for r in ledger.records])
    say(f"   {len(ledger.records)} records -> {len(chain.blocks)} blocks (Merkle root per block, proof of work, each block links to the last)")
    for b in chain.blocks[-3:]:
        say(f"     block {b.index:<3} {len(b.records)} records  merkle {b.merkle[:10]}...  nonce {b.nonce:<6} hash {b.hash[:12]}...")
    say(f"   chain verify: {chain.verify()[2]}")
    if victim is not None:
        net = Network(chain, 3)
        say(f"\n   Attack 1: one replica edits record #{victim['seq']} (blame the LDR) and nothing else")
        net.edit_record("node-B", victim["seq"], lambda r: r["data"].update(sensor="ldr"))
        okb, blk, why = net.node("node-B").chain.verify()
        say(f"   -> node-B's own check: {'passed' if okb else 'FAILED at block %s (%s)' % (blk, why)}")
        net = Network(chain, 3)
        say(f"\n   Attack 2: a skilled attacker rewrites the same record on node-B and recomputes every hash and block")
        net.rewrite_history("node-B", victim["seq"], lambda r: r["data"].update(sensor="ldr"))
        say(f"   -> node-B's own check: {'PASSED (a lone hash chain is fooled)' if net.node('node-B').chain.verify()[0] else 'failed'}")
        vote = net.vote()
        if vote.dissenters:
            who, blk = next(iter(vote.dissenters.items()))
            say(f"   -> majority vote: REJECTED {who}, its block {blk} differs from the other two; "
                f"{vote.agreed_blocks}/{vote.total_blocks} blocks agreed")
        else:
            say("   !! the vote did not notice")
        say(f"   -> repaired from the majority: {', '.join(net.heal()) or 'nothing'}; vote now {'clean' if net.vote().clean else 'NOT clean'}")
    say("   (replicas are simulated in one process; a real deployment would put them on separate machines)")

    # ---- summary ----------------------------------------------------------------------------------------------------
    banner("9", "SUMMARY")
    say(f"   {'window labels':<22}{'TRUSTED':>10}{'UNCERTAIN':>11}{'SUSPICIOUS':>12}")
    for p, c in counts.items():
        say(f"   {p:<22}{c['TRUSTED']:>10}{c['UNCERTAIN']:>11}{c['SUSPICIOUS']:>12}")
    q = [e for e in pipe.security.events if e.kind == "sensor_quarantined"]
    say(f"\n   attack injected into the PIR for {a.attack_hours:g} h; "
        + (f"quarantined at device hour {q[0].t / 3600:.1f}" if q else "NOT quarantined (the attack may be too short or the room not empty)"))
    say(f"   recovery: {'completed' if ctx['recovered'] else 'not completed (recording ended first, or the sensor was never released)'}")
    say(f"   ledger: {'intact' if ok else 'BROKEN'}   final sensor states: "
        + ", ".join(f"{s} {v.value}" for s, v in pipe.security.state.items()))
    if not a.no_export:
        out = Path(a.export_dir) if a.export_dir else HERE / "exports" / stamp
        meta = {
            "title": "demo_pipeline run", "recording": str(a.replay), "input": st.summary(),
            "preset": f"demo x{a.demo:g}" if a.demo else "room", "demo_factor": a.demo,
            "calibration": ctx.get("cal_text"), "calibration_end_h": round(ctx["cal_t"] / 3600.0, 3) if ctx.get("cal_t") is not None else None,
            "attack": {"sensor": "pir", "forced_to": 1, "start_h": a.attack_start_h, "hours": a.attack_hours, "software_only": True},
            "release_after_h": a.release_after_h, "window_s": cfg.features.window_s,
            "threshold": cfg.anomaly.mahalanobis_threshold,
            "counts": {k: dict(v) for k, v in counts.items()},
            "quarantined_at_h": round(q[0].t / 3600.0, 3) if q else None, "recovered": ctx["recovered"],
            "ledger_intact": ok, "final_states": {s_: v.value for s_, v in pipe.security.state.items()},
            "ledger_records": len(ledger.records),
        }
        try:
            saved = rec.save(out, ledger, ledger_path, meta, a.replay)
            say(f"\n   run saved for the dashboard: {saved}")
        except Exception as e:      # the demo itself must never fail because an export could not be written
            say(f"\n   (run files were not saved: {e})")
    return 0 if q else 3


if __name__ == "__main__":
    sys.exit(main())