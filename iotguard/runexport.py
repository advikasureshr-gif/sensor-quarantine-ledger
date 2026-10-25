"""
runexport.py - save one demo_pipeline run as plain files (CSV / JSON / JSONL) so a dashboard or a report can load it.

    exports/<stamp>/
        meta.json                what was run: recording, preset, attack, counts, outcome
        windows.csv              one row per operating window: label, learning rate, score, drift, muted sensors ...
        events.json              alerts, quarantine, release, recovery, and the run's milestones (attack start/end ...)
        ledger.jsonl             the hash-chained event ledger (a copy)
        chain.json               the ledger as blocks (Merkle roots, nonces, hashes) + the replica-vote demonstration
        recording_summary.json   what the raw recording looked like (hours, PIR / door / LDR / DHT statistics)
        recording_series.csv     the raw recording averaged per minute (PIR, door, LDR, temperature, humidity)

Nothing here changes what the detector decides; it only writes down what the run produced.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional

from .chain import Blockchain, Network
from .security import Ledger

MODE = {"TRUSTED": "full", "UNCERTAIN": "slow", "SUSPICIOUS": "freeze"}
WINDOW_FIELDS = ["t_h", "index", "period", "label", "mode", "alpha", "score", "drift", "naive_drift", "divergence",
                 "occupancy", "suspicion", "muted", "contradictions", "reason"]
SERIES_FIELDS = ["t_h", "pir", "door", "ldr", "temp", "hum"]


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


class RunRecorder:
    """Collects rows while a pipeline runs; `save()` writes the files."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.marks: list[dict] = []

    def window(self, o, period: str) -> None:
        d = o.decision
        if d is None:
            return
        self.rows.append({
            "t_h": _num(o.t_start / 3600.0), "index": o.index, "period": period, "label": d.label.value,
            "mode": MODE.get(d.label.value, ""), "alpha": _num(o.alpha), "score": _num(o.score), "drift": _num(o.drift),
            "naive_drift": _num(o.naive_drift), "divergence": _num(o.divergence),
            "occupancy": getattr(getattr(d, "occupancy", None), "value", ""), "suspicion": _num(d.suspicion),
            "muted": ",".join(o.muted), "contradictions": ",".join(o.consensus.contradictions) if o.consensus else "",
            "reason": d.reason,
        })

    def event(self, e) -> None:
        self.events.append({"kind": e.kind, "t_h": _num(e.t / 3600.0), "sensor": e.sensor, "detail": e.detail})

    def mark(self, kind: str, t: float, detail: str = "") -> None:
        self.marks.append({"kind": kind, "t_h": _num(t / 3600.0), "detail": detail})

    def save(self, out_dir, ledger: Ledger, ledger_path, meta: dict, recording_path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "windows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in WINDOW_FIELDS})
        led_events = [{"kind": r["event"], "t_h": _num(r["t"] / 3600.0) if r.get("t") is not None else None,
                       "sensor": r["data"].get("sensor"), "detail": r["data"].get("detail", "")}
                      for r in ledger.records if r["event"] in ("auth_failed", "auth_locked", "auth_refused_locked")]
        (out / "events.json").write_text(json.dumps({"events": self.events + led_events, "marks": self.marks}, indent=1), encoding="utf-8")
        if ledger_path and Path(ledger_path).exists():
            shutil.copyfile(ledger_path, out / "ledger.jsonl")
        else:
            (out / "ledger.jsonl").write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in ledger.records),
                                              encoding="utf-8")
        (out / "chain.json").write_text(json.dumps(chain_summary([dict(r) for r in ledger.records]), indent=1), encoding="utf-8")
        summary, series = summarize_recording(recording_path)
        (out / "recording_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        with (out / "recording_series.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SERIES_FIELDS)
            w.writeheader()
            w.writerows(series)
        meta = dict(meta, saved=time.strftime("%Y-%m-%dT%H:%M:%S"), n_windows=len(self.rows))
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# the blockchain view of a ledger
# ---------------------------------------------------------------------------
def chain_summary(records: list[dict]) -> dict:
    """Blocks (no record bodies) plus what happens when one replica is attacked, for display and for the report."""
    if not records:
        return {"blocks": [], "valid": True, "why": "empty ledger", "demo": None}
    chain = Blockchain.build(records)
    ok, _, why = chain.verify()
    out = {
        "difficulty": chain.blocks[0].difficulty, "block_size": len(chain.blocks[0].records), "valid": ok, "why": why,
        "blocks": [{"index": b.index, "n_records": len(b.records), "first_seq": b.records[0]["seq"], "last_seq": b.records[-1]["seq"],
                    "timestamp": b.timestamp, "merkle": b.merkle, "nonce": b.nonce, "prev_hash": b.prev_hash, "hash": b.hash}
                   for b in chain.blocks],
        "demo": None,
    }
    victim = next((r for r in records if r["event"] == "sensor_quarantined"), records[len(records) // 2])
    seq = victim["seq"]

    def blame_ldr(r):
        r["data"]["sensor"] = "ldr"

    n1 = Network(chain, 3)
    n1.edit_record("node-B", seq, blame_ldr)
    ok1, blk1, why1 = n1.node("node-B").chain.verify()
    n2 = Network(chain, 3)
    n2.rewrite_history("node-B", seq, blame_ldr)
    own_ok = n2.node("node-B").chain.verify()[0]
    v = n2.vote()
    healed = n2.heal()
    out["demo"] = {
        "record_seq": seq, "record_event": victim["event"],
        "naive_edit": {"caught_by_node_itself": not ok1, "block": blk1, "why": why1},
        "skilled_rewrite": {"passes_own_check": own_ok, "rejected_by_vote": bool(v.dissenters), "dissenters": v.dissenters,
                            "agreed_blocks": v.agreed_blocks, "total_blocks": v.total_blocks, "healed": healed,
                            "vote_clean_after_heal": n2.vote().clean},
    }
    return out


# ---------------------------------------------------------------------------
# the raw recording
# ---------------------------------------------------------------------------
def summarize_recording(path, bucket_s: float = 60.0) -> tuple[dict, list[dict]]:
    """Statistics and a per-minute series for a recording made by iotguard.record (never alters the file)."""
    n = nulls = bad = 0
    t0 = tlast = None
    pir_high = pir_edges = door_changes = door_open = 0
    longest_high = 0.0
    high_since = None
    last_pir = last_door = None
    ldr = {"0": 0, "4095": 0, "other": 0}
    temps, hums = [], []
    gaps, max_gap = 0, 0.0
    series: list[dict] = []
    acc = None

    def flush():
        nonlocal acc
        if acc and acc["n"]:
            series.append({"t_h": round(acc["t"] / 3600.0, 5), "pir": round(acc["pir"] / acc["n"], 4), "door": acc["door"],
                           "ldr": round(acc["ldr"] / acc["n"], 1),
                           "temp": round(acc["temp"] / acc["nt"], 2) if acc["nt"] else "",
                           "hum": round(acc["hum"] / acc["nt"], 2) if acc["nt"] else ""})
        acc = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
                t = float(r["t"])
                pir, door, ld = int(r["pir"]), int(r["door"]), int(r["ldr"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            t0 = t if t0 is None else t0
            if tlast is not None:
                dt = t - tlast
                if dt > 5.0:
                    gaps += 1
                    max_gap = max(max_gap, dt)
            tlast = t
            n += 1
            pir_high += pir
            door_open += 1 if door else 0
            if last_pir is not None and pir and not last_pir:
                pir_edges += 1
                high_since = t
            if pir and high_since is None:
                high_since = t
            if not pir and high_since is not None:
                longest_high = max(longest_high, t - high_since)
                high_since = None
            if last_door is not None and door != last_door:
                door_changes += 1
            last_pir, last_door = pir, door
            ldr["0" if ld == 0 else "4095" if ld == 4095 else "other"] += 1
            if r.get("temp") is None or r.get("hum") is None:
                nulls += 1
            else:
                temps.append(float(r["temp"]))
                hums.append(float(r["hum"]))
            b = int((t - t0) // bucket_s)
            if acc is None or acc["b"] != b:
                flush()
                acc = {"b": b, "t": t - t0, "n": 0, "nt": 0, "pir": 0, "door": door, "ldr": 0, "temp": 0.0, "hum": 0.0}
            acc["n"] += 1
            acc["pir"] += pir
            acc["door"] = max(acc["door"], door)
            acc["ldr"] += ld
            if r.get("temp") is not None and r.get("hum") is not None:
                acc["nt"] += 1
                acc["temp"] += float(r["temp"])
                acc["hum"] += float(r["hum"])
    flush()
    if high_since is not None and tlast is not None:
        longest_high = max(longest_high, tlast - high_since)
    hours = ((tlast - t0) / 3600.0) if n else 0.0
    quiet = []
    try:
        from .inject import quiet_stretches
        from .sources import ReplaySource
        stretches, _ = quiet_stretches(ReplaySource(str(path)))
        quiet = [{"start_h": round(s, 2), "end_h": round(e, 2), "hours": round(e - s, 2)}
                 for s, e in sorted(stretches, key=lambda x: x[0] - x[1])[:3]]
    except Exception:
        pass
    summary = {
        "file": Path(path).name, "readings": n, "bad_lines": bad, "hours": round(hours, 3),
        "dht_null": nulls, "dht_null_pct": round(100.0 * nulls / n, 3) if n else 0.0,
        "pir_high_pct": round(100.0 * pir_high / n, 2) if n else 0.0, "pir_triggers": pir_edges,
        "pir_longest_high_s": round(longest_high, 1),
        "door_changes": door_changes, "door_open_pct": round(100.0 * door_open / n, 2) if n else 0.0,
        "ldr_0_pct": round(100.0 * ldr["0"] / n, 2) if n else 0.0, "ldr_4095_pct": round(100.0 * ldr["4095"] / n, 2) if n else 0.0,
        "ldr_other_pct": round(100.0 * ldr["other"] / n, 2) if n else 0.0,
        "gaps_over_5s": gaps, "max_gap_s": round(max_gap, 1),
        "temp_min": min(temps) if temps else None, "temp_max": max(temps) if temps else None,
        "hum_min": min(hums) if hums else None, "hum_max": max(hums) if hums else None,
        "quietest_stretches": quiet,
    }
    return summary, series