"""Admin gate, the quarantine dialog's engine action, and the live/replay runner."""
import itertools
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components.runner import EngineRun                      # noqa: E402
from iotguard.security import AdminAuth, AuthError, SensorState   # noqa: E402

SECRET = "test-admin-secret"


def make_run(tmp_path, kind="live"):
    return EngineRun(kind, "t", AdminAuth(SECRET, iterations=1000), SECRET, demo=2.0, sessions_dir=tmp_path)


def test_wrong_secret_changes_nothing_and_is_logged(tmp_path):
    run = make_run(tmp_path)
    run.pipe.security.state["pir"] = SensorState.QUARANTINED
    with pytest.raises(AuthError):
        run.admin_action("pir", "replace", "wrong-secret")
    assert run.quarantined() == ["pir"]
    ev = [r["event"] for r in run.ledger_records()]
    assert "auth_failed" in ev and "sensor_replaced" not in ev


def test_replace_and_unquarantine_start_probation(tmp_path):
    for action, want in (("replace", "sensor_replaced"), ("unquarantine", None)):
        run = make_run(tmp_path)
        run.pipe.security.state["pir"] = SensorState.QUARANTINED
        run.admin_action("pir", action, SECRET, "new unit")
        assert run.sensor_states()["pir"] == "PROBATION"
        ev = [r["event"] for r in run.ledger_records()]
        assert "sensor_probation" in ev and (want is None or want in ev)
        assert run.ledger.verify()[0]


def test_lockout_after_three_failures(tmp_path):
    run = make_run(tmp_path)
    run.pipe.security.state["pir"] = SensorState.QUARANTINED
    for _ in range(3):
        with pytest.raises(AuthError):
            run.admin_action("pir", "unquarantine", "nope-nope")
    with pytest.raises(AuthError):
        run.admin_action("pir", "unquarantine", SECRET)          # locked out even with the right secret
    assert "auth_locked" in [r["event"] for r in run.ledger_records()]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    from iotguard.scenarios import make_source
    from iotguard.sources import strip_labels, write_jsonl
    p = tmp_path_factory.mktemp("rec") / "sim.jsonl"
    write_jsonl(strip_labels(make_source("normal", 1, days=2.1)), str(p), include_labels=False)
    return p


def test_live_runner_learns_then_follows_the_board(tmp_path, recording):
    from iotguard.sources import ReplaySource
    run = make_run(tmp_path)
    run.start_live(str(recording), lambda: itertools.islice(ReplaySource(str(recording)), 3000))
    run.thread.join(timeout=120)
    assert run.status == "stopped" and run.error is None, (run.status, run.error)
    df = run.frame()
    assert len(df) >= 40 and run.t0_h > 20 and df["t_h"].min() >= run.t0_h
    assert set(df["label"]) <= {"TRUSTED", "UNCERTAIN", "SUSPICIOUS"}
    assert run.latest is not None and run.ledger.verify()[0]
    assert run.log_path.exists() and run.log_path.parent.name == "live_logs" and run.log_lines == 3000
    from iotguard.sources import ReplaySource as RS
    assert sum(1 for _ in RS(str(run.log_path))) == 3000          # the saved file replays like any recording


def test_live_software_attack_freezes_learning(tmp_path, recording):
    from iotguard.sources import ReplaySource
    run = make_run(tmp_path)
    run.attack_on = True
    run.start_live(str(recording), lambda: itertools.islice(ReplaySource(str(recording)), 4000))
    run.thread.join(timeout=120)
    df = run.frame()
    assert (df["label"] == "SUSPICIOUS").sum() >= 5 and df[df["label"] == "SUSPICIOUS"]["alpha"].max() == 0.0
    assert any(m["kind"] == "attack_start" for m in run.marks)


def test_admin_gate_on_pages(tmp_path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages" / "9_Admin.py"), default_timeout=60).run()
    assert not at.exception and any("Sign in" in m.value for m in at.markdown)
    assert not at.tabs                                           # nothing of the ledger is drawn for a visitor
    at.session_state["_admin_until"] = time.time() + 600
    at.run()
    assert not at.exception
