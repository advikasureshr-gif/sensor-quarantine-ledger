"""The 'Real Recording Run' page loads a saved run and its replica buttons behave. Uses a tiny hand-made run folder."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components import real_run as rr          # noqa: E402
from iotguard.security import Ledger            # noqa: E402


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    led = Ledger(d / "ledger.jsonl", clock=lambda: "2026-10-24T10:00:00")
    for k in range(9):
        led.append("sensor_quarantined" if k == 5 else "baseline_checkpoint", float(k * 3600), sensor="pir", detail="x")
    rows = []
    for i in range(300):
        t = 10 + i / 120.0
        period = "before the attack" if t < 11.0 else "during the attack" if t < 12.0 else "after the attack"
        mode = "freeze" if period == "during the attack" and i % 2 else "full"
        rows.append({"t_h": t, "index": i, "period": period, "label": "SUSPICIOUS" if mode == "freeze" else "TRUSTED", "mode": mode,
                     "alpha": 0.0 if mode == "freeze" else 0.002, "score": 2.0 + (3 if mode == "freeze" else 0), "drift": 0.01 * i / 300,
                     "naive_drift": 0.02 * i / 300, "divergence": 0.0, "occupancy": "EMPTY", "suspicion": 0.0,
                     "muted": "pir" if period == "after the attack" and i % 3 == 0 else "", "contradictions": "", "reason": "consistent"})
    pd.DataFrame(rows).to_csv(d / "windows.csv", index=False)
    (d / "events.json").write_text(json.dumps({"events": [{"kind": "alert", "t_h": 11.2, "sensor": None, "detail": "x"},
                                                         {"kind": "sensor_quarantined", "t_h": 11.9, "sensor": "pir", "detail": "y"}],
                                               "marks": [{"kind": "attack_start", "t_h": 11.0, "detail": ""}, {"kind": "attack_end", "t_h": 12.0, "detail": ""},
                                                         {"kind": "admin_release", "t_h": 12.4, "detail": ""}]}), encoding="utf-8")
    (d / "chain.json").write_text("{}", encoding="utf-8")
    (d / "recording_summary.json").write_text(json.dumps({
        "file": "room.jsonl", "readings": 36000, "bad_lines": 0, "hours": 10.0, "dht_null": 0, "dht_null_pct": 0.0, "pir_high_pct": 1.0, "pir_triggers": 5,
        "pir_longest_high_s": 20.0, "door_changes": 4, "door_open_pct": 0.5, "ldr_0_pct": 50.0, "ldr_4095_pct": 49.5, "ldr_other_pct": 0.5,
        "gaps_over_5s": 0, "max_gap_s": 1.0, "temp_min": 26.0, "temp_max": 28.0, "hum_min": 70.0, "hum_max": 80.0,
        "quietest_stretches": [{"start_h": 1.0, "end_h": 6.0, "hours": 5.0}]}), encoding="utf-8")
    pd.DataFrame({"t_h": [i / 60 for i in range(0, 600, 5)], "pir": 0.0, "door": 0, "ldr": 4095.0, "temp": 27.0, "hum": 75.0}).to_csv(d / "recording_series.csv", index=False)
    (d / "meta.json").write_text(json.dumps({
        "recording": "logs/room.jsonl", "preset": "demo x2", "calibration_end_h": 10.0, "threshold": 4.0, "quarantined_at_h": 11.9, "recovered": True,
        "ledger_intact": True, "attack": {"sensor": "pir", "forced_to": 1, "start_h": 11.0, "hours": 1.0}}), encoding="utf-8")
    return d


def test_loader_and_helpers(run_dir):
    assert rr.is_run(run_dir) and not rr.is_run(run_dir.parent)
    run = rr.load_run(run_dir)
    assert run.attack_span == (11.0, 12.0) and run.mark_h("admin_release") == 12.4
    t = rr.mode_table(run)
    assert list(t.index) == ["before the attack", "during the attack", "after the attack"] and t.loc["during the attack", "freeze"] > 0
    assert rr.timeline_figure(run) is not None and rr.mode_figure(run) is not None and rr.recording_figure(run) is not None
    notes = dict(rr.recording_notes(run.rec) and [(k, v) for k, v in rr.recording_notes(run.rec)])
    assert "warn" not in notes
    with pytest.raises(FileNotFoundError):
        rr.load_run(run_dir.parent)


def test_page_renders_and_replica_buttons_work(run_dir):
    from streamlit.testing.v1 import AppTest
    import time
    at = AppTest.from_file(str(ROOT / "pages" / "6_Real_Recording_Run.py"), default_timeout=120)
    at.run()
    at.sidebar.text_input[0].set_value(str(run_dir)).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(b.label.startswith("Attack 1") for b in at.button)            # a visitor sees a lock, not the ledger
    assert any("admin-only" in m.value for m in at.markdown)
    at.session_state["_admin_until"] = time.time() + 600
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Timeline", "Learning rate", "Events", "Ledger & blockchain", "Recording logs"]

    def replicas():
        return next(d.value for d in at.dataframe if "replica" in d.value.columns).set_index("replica")

    next(b for b in at.button if b.label.startswith("Attack 1")).click().run()
    assert "FAILS" in replicas().loc["node-B", "own check"] and "DISAGREES" in replicas().loc["node-B", "majority vote"]
    next(b for b in at.button if b.label.startswith("Repair")).click().run()
    next(b for b in at.button if b.label.startswith("Attack 2")).click().run()
    assert replicas().loc["node-B", "own check"] == "passes" and "DISAGREES" in replicas().loc["node-B", "majority vote"]
    next(b for b in at.button if b.label.startswith("Repair")).click().run()
    assert set(replicas()["majority vote"]) == {"agrees"} and not at.exception


def test_page_shows_empty_state_without_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rr, "find_runs", lambda: [])
    at = AppTest.from_file(str(ROOT / "pages" / "6_Real_Recording_Run.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No saved run yet" in m.value for m in at.markdown)
