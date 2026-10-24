"""Basic tests for dashboard_service. Run from the package folder:  python -m pytest tests -q
The one test marked `slow` runs the REAL engine for a 2.5-day scenario (about 30-60 s)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_service as ds  # noqa: E402


def test_catalog_matches_engine():
    from iotguard.scenarios import SCENARIOS
    assert {c["name"] for c in ds.scenario_catalog()} == set(SCENARIOS)


@pytest.mark.parametrize("kw", [dict(scenario="nope"), dict(scenario="normal", seed=-1), dict(scenario="normal", seed=True),
                                dict(scenario="normal", days=1.0), dict(scenario="normal", days=99), dict(scenario="normal", strength=0),
                                dict(scenario="normal", profile="x"), dict(scenario="pir_stuck_night", days=3.0)])
def test_invalid_params_rejected(kw):
    with pytest.raises(ds.ParamError):
        ds.validate_params(**kw)


def test_valid_params_ok():
    p = ds.validate_params("temp_spike", 5, 4.0, 1.5, "dht11")
    assert p.scenario == "temp_spike" and p.seed == 5


def test_ground_truth_windows():
    m = {"attacks": [{"tag": "a", "start_s": 100, "end_s": 200}], "legit_changes": [{"kind": "k", "start_s": 300, "label_end_s": None}]}
    assert ds.ground_truth_for_window(m, 90, 120)[0] == "attack"
    assert ds.ground_truth_for_window(m, 200, 230)[0] == "normal"
    assert ds.ground_truth_for_window(m, 400, 430)[0] == "legit_change"


def _frame(decisions, truths):
    return pd.DataFrame({"phase": "OPERATING", "valid": True, "decision": decisions, "truth": truths})


def test_evaluate_counts_and_undefined():
    m = {"attacks": [1]}
    ev = ds.evaluate(_frame(["SUSPICIOUS", "TRUSTED", "SUSPICIOUS", "TRUSTED"], ["attack", "attack", "normal", "normal"]), m)["strict"]
    assert (ev["tp"], ev["fn"], ev["fp"], ev["tn"]) == (1, 1, 1, 1) and ev["precision"] == 0.5 and ev["recall"] == 0.5
    none = ds.evaluate(_frame(["TRUSTED", "TRUSTED"], ["normal", "normal"]), m)["strict"]
    assert none["recall"] is None and none["precision"] is None and none["fpr"] == 0.0
    assert ds.evaluate(pd.DataFrame(columns=ds.WINDOW_COLUMNS), m)["available"] is False
    assert ds.evaluate(_frame(["TRUSTED"], ["normal"]), {})["available"] is False


def test_ledger_tamper_detected():
    from iotguard.security import Ledger
    led = Ledger()
    for i in range(3):
        led.append("x", float(i), n=i)
    recs = ds.json_safe(led.records)
    assert ds.verify_records(recs) == (True, None)
    recs[1]["data"]["n"] = 99
    assert ds.verify_records(recs)[0] is False


def test_manager_refuses_concurrent_run(tmp_path):
    import threading
    gate = threading.Event()
    mgr = ds.RunManager(tmp_path, executor=lambda rec: (setattr(rec, "status", "running"), gate.wait(5), setattr(rec, "status", "completed")))
    p = ds.validate_params("normal", 0, 2.5)
    mgr.start(p)
    with pytest.raises(ds.RunBusyError):
        mgr.start(p)
    gate.set()
    mgr.wait(5)


@pytest.mark.slow
def test_real_run_end_to_end(tmp_path):
    mgr = ds.RunManager(tmp_path)
    rec = mgr.start(ds.validate_params("normal", 0, 2.5))
    mgr.wait(600)
    assert rec.status == "completed", rec.error
    assert rec.summary["windows_processed"] == len(rec.windows) > 0
    assert rec.summary["phase"] == "OPERATING" and rec.summary["ledger_chain_ok"] is True
    again = ds.RunManager(tmp_path).runs[rec.run_id]                 # saved to disk and reloaded
    assert len(again.windows) == len(rec.windows) and ds.verify_records(again.ledger)[0]
