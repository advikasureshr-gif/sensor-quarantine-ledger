"""Saving and restoring the whole system: nothing a restart must keep may be lost, and tampering must be detected."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from iotguard.conftest import feed, key
from iotguard.config import DEFAULT_CONFIG, OnlineStatsConfig
from iotguard.pipeline import Phase, Pipeline, StateIntegrityError, StateMismatchError
from iotguard.security import AdminAuth, Ledger, SensorState

WINDOW = 60                     # readings per 30 s window at 2 Hz


def calibrated(small_cfg, new_consensus, stream, n_windows, **kw):
    """A pipeline that has processed exactly `n_windows` windows (feeding one extra reading to complete the last)."""
    p = Pipeline(small_cfg, consensus=new_consensus, **kw)
    p.start_calibration()
    feed(p, stream[: WINDOW * n_windows + 1])
    return p


def test_resume_matches_uninterrupted_run(tmp_path, small_cfg, new_consensus, stream):
    n = 500                                                   # 4.2 h: calibrated at ~1 h, then operating
    a = calibrated(small_cfg, new_consensus, stream, n)
    assert a.phase is Phase.OPERATING
    a.save(tmp_path / "s.npz")
    b = Pipeline.load(tmp_path / "s.npz", small_cfg, new_consensus)
    assert b.phase is Phase.OPERATING and b.model is not None
    assert np.array_equal(a.model.mu, b.model.mu) and np.array_equal(a.model.sigma, b.model.sigma)
    rest_a = feed(a, stream[WINDOW * n + 1:])
    rest_b = feed(b, stream[WINDOW * n:])                     # b never saw reading 60n (it was in a's unfinished window)
    ka, kb = key(rest_a), key(rest_b)
    assert len(ka) == len(kb) > 100
    assert [k[:2] for k in ka] == [k[:2] for k in kb], "same windows, same phases"
    # The preprocessor restarts cold, so the first window or two differ slightly, and because the model learns from every
    # window that difference persists as tiny numerical drift. Decisions must be identical; scores must stay close.
    labels_a, labels_b = [k[2] for k in ka], [k[2] for k in kb]
    assert labels_a == labels_b, [i for i, (x, y) in enumerate(zip(labels_a, labels_b)) if x != y]
    diff = np.abs(np.array([k[3] for k in ka]) - np.array([k[3] for k in kb]))
    assert diff[3:].max() < 0.02 and diff[:3].max() < 1.0, (diff[:5], diff.max())


def test_restart_during_calibration_keeps_collected_data(tmp_path, small_cfg, new_consensus, stream):
    ref = Pipeline(small_cfg, consensus=new_consensus); ref.start_calibration(); out_ref = feed(ref, stream)
    fin_ref = [o for o in out_ref if o.calibration is not None][0].calibration
    a = calibrated(small_cfg, new_consensus, stream, 60)                          # 30 minutes into a 1 hour calibration
    assert a.phase is Phase.CALIBRATING and a.calibrator.n_candidates >= 55
    a.save(tmp_path / "s.npz")
    b = Pipeline.load(tmp_path / "s.npz", small_cfg, new_consensus)
    assert b.phase is Phase.CALIBRATING and b.calibrator.n_candidates == a.calibrator.n_candidates
    out_b = feed(b, stream[WINDOW * 60:])
    fin_b = [o for o in out_b if o.calibration is not None][0].calibration
    assert fin_b.status.value == "READY" and abs(fin_b.n_trusted - fin_ref.n_trusted) <= 2 and abs(fin_b.n_candidates - fin_ref.n_candidates) <= 2
    assert np.allclose(fin_b.model.mu, fin_ref.model.mu, rtol=0.02, atol=0.05)


def test_quarantine_survives_restart(tmp_path, small_cfg, new_consensus, attack_stream):
    auth = AdminAuth("admin secret", iterations=1000)
    a = Pipeline(small_cfg, consensus=new_consensus, auth=auth); a.start_calibration("admin secret")
    cut = 2 * 3600 * 7                                                             # 7 h of data = 4 h into the attack
    feed(a, attack_stream[: cut + 1])
    assert a.security.state["pir"] is SensorState.QUARANTINED, "setup: the stuck PIR should be quarantined by now"
    a.save(tmp_path / "s.npz")
    b = Pipeline.load(tmp_path / "s.npz", small_cfg, new_consensus, auth=AdminAuth("admin secret", iterations=1000))
    assert b.security.state["pir"] is SensorState.QUARANTINED and b.security.muted_features() == ("pir_count", "pir_duty")
    out = feed(b, attack_stream[cut: cut + 1200])
    assert out and all("pir" in o.muted for o in out if o.decision is not None), "a restart must never clear a quarantine"
    # and it behaves as if there had been no restart
    ref = feed(a, attack_stream[cut + 1: cut + 1200])
    la = [o.decision.label.value for o in ref if o.decision is not None]
    lb = [o.decision.label.value for o in out if o.decision is not None]
    assert la == lb[:len(la)]
    b.release_sensor("pir", "admin secret", 1.0)                                  # the admin can still act after a restart
    assert b.security.state["pir"] is SensorState.PROBATION


def test_clock_restart_is_handled(tmp_path, small_cfg, new_consensus, stream):
    """The ESP32 reboots while the host restarts too: the device clock starts again from 0."""
    n = 500
    a = calibrated(small_cfg, new_consensus, stream, n)
    a.save(tmp_path / "s.npz")
    b = Pipeline.load(tmp_path / "s.npz", small_cfg, new_consensus)
    t0 = stream[WINDOW * n].t
    shifted = [replace(r, t=round(r.t - t0 + 5.0, 3)) for r in stream[WINDOW * n: WINDOW * n + 2400]]
    out = feed(b, shifted)
    ops = [o for o in out if o.decision is not None]
    assert len(ops) >= 30 and all(np.isfinite(o.score) for o in ops)
    assert sum(o.decision.label.value == "SUSPICIOUS" for o in ops) <= 2
    # re-anchored ages are sane: the last independent evidence is not in the future and not absurdly old
    eng = b.engine
    assert eng._t_ind <= ops[-1].t_start + 60.0


def test_tampering_with_the_saved_file_is_detected(tmp_path, small_cfg, new_consensus, stream):
    ledger = Ledger(clock=lambda: "T")
    a = calibrated(small_cfg, new_consensus, stream, 200, ledger=ledger)
    path = tmp_path / "s.npz"; a.save(path)
    Pipeline.load(path, small_cfg, new_consensus, ledger=ledger)                  # untouched: loads fine
    # rewrite the file with a doctored baseline mean
    z = dict(np.load(path, allow_pickle=False)); z["model_mu"] = z["model_mu"] + 1.0
    with open(path, "wb") as f:
        np.savez(f, **z)
    with pytest.raises(StateIntegrityError):
        Pipeline.load(path, small_cfg, new_consensus, ledger=ledger)


def test_altered_registered_baseline_is_detected_even_without_a_ledger(tmp_path, small_cfg, new_consensus, stream):
    a = calibrated(small_cfg, new_consensus, stream, 200); path = tmp_path / "s.npz"; a.save(path)
    z = dict(np.load(path, allow_pickle=False)); z["reg1_mu"] = z["reg1_mu"] * 1.01
    with open(path, "wb") as f:
        np.savez(f, **z)
    with pytest.raises(ValueError, match="altered"):
        Pipeline.load(path, small_cfg, new_consensus)


def test_different_settings_are_refused(tmp_path, small_cfg, new_consensus, stream):
    a = calibrated(small_cfg, new_consensus, stream, 200); path = tmp_path / "s.npz"; a.save(path)
    other = replace(small_cfg, online_stats=replace(small_cfg.online_stats, alpha_trusted=0.01))
    with pytest.raises(StateMismatchError):
        Pipeline.load(path, other, new_consensus)
    with pytest.raises(StateMismatchError):
        Pipeline.load(path, small_cfg, replace(new_consensus, vacancy_timeout_s=300.0))


def test_long_downtime_resets_occupancy_but_keeps_everything_else(tmp_path, small_cfg, new_consensus, stream):
    a = calibrated(small_cfg, new_consensus, stream, 500); path = tmp_path / "s.npz"
    a.engine.state = a.engine.state.__class__("OCCUPIED"); a.engine._conf = 1.0           # believed occupied when it stopped
    a.gate.suspicion = 2.0
    saved_wall = a._wall(); a.save(path)
    quick = Pipeline.load(path, small_cfg, new_consensus, wall_now=saved_wall + 60)
    assert quick.engine.state.value == "OCCUPIED" and quick.gate.suspicion == 2.0
    slow = Pipeline.load(path, small_cfg, new_consensus, wall_now=saved_wall + 7200)
    assert slow.engine.state.value == "UNKNOWN" and slow.gate.suspicion == 0.0
    assert np.array_equal(slow.model.mu, a.model.mu) and slow.engine.light_enforced == a.engine.light_enforced


def test_save_is_atomic_and_pickle_free(tmp_path, small_cfg, new_consensus, stream, monkeypatch):
    a = calibrated(small_cfg, new_consensus, stream, 200); path = tmp_path / "s.npz"; a.save(path)
    before = path.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    np.load(path, allow_pickle=False).files                                          # a pickle-free archive loads without allow_pickle
    def boom(*args, **kw): raise OSError("disk full")
    monkeypatch.setattr(np, "savez", boom)
    feed(a, stream[WINDOW * 200 + 1: WINDOW * 220])
    with pytest.raises(OSError):
        a.save(path)
    assert path.read_bytes() == before, "a failed save must leave the previous file intact"


def test_autosave_and_save_on_interrupt(tmp_path, small_cfg, new_consensus, stream):
    ledger = Ledger(clock=lambda: "T")
    p = Pipeline(small_cfg, consensus=new_consensus, ledger=ledger, state_path=tmp_path / "s.npz", autosave_every_windows=40)
    p.start_calibration()
    def interrupted():
        for i, r in enumerate(stream):
            if i == WINDOW * 150 + 5:
                raise KeyboardInterrupt
            yield r
    with pytest.raises(KeyboardInterrupt):
        p.run(interrupted())
    saves = ledger.events("state_saved")
    assert len(saves) >= 4, "autosaves every 40 windows plus the save on interrupt"
    q = Pipeline.load(tmp_path / "s.npz", small_cfg, new_consensus, ledger=ledger)
    assert q.phase is Phase.OPERATING and q.n_windows == p.n_windows
    assert ledger.verify()[0]