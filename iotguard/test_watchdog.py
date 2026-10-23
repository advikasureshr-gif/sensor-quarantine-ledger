"""The hardware watchdog: reports dead or frozen sensors and stream gaps, never false alarms on a healthy room,
and never alters a detection decision."""
from dataclasses import replace

from iotguard.conftest import feed, key
from iotguard.contract import Reading
from iotguard.pipeline import Pipeline
from iotguard.run import _main
from iotguard.scenarios import profile_configs
from iotguard.security import Ledger
from iotguard.sim_sensors import SimulatorSource
from iotguard.sim_world import World
from iotguard.watchdog import HealthConfig, Watchdog


def rd(t, temp=22.0, hum=50.0, ldr=1500, pir=0, door=0):
    return Reading(t=t, temp=temp, hum=hum, pir=pir, door=door, ldr=ldr)


def run(w, readings):
    out = []
    for r in readings:
        out += w.observe(r)
    return out


def test_dht_dead_only_after_the_limit_and_clears():
    w = Watchdog()
    ev = run(w, [rd(t * 0.5, temp=None, hum=None) for t in range(0, 500)])          # 250 s of nulls: still fine
    assert ev == []
    ev = run(w, [rd(250 + t * 0.5, temp=None, hum=None) for t in range(0, 200)])    # passes 300 s
    assert [(e.kind, e.fault) for e in ev] == [("sensor_fault", "dht_dead")]
    ev = run(w, [rd(350.0)])
    assert [(e.kind, e.fault) for e in ev] == [("sensor_fault_cleared", "dht_dead")] and not w.active_faults()


def test_one_missing_value_is_not_a_dead_dht():
    w = Watchdog()
    assert run(w, [rd(0.5, temp=None, hum=22.0)] * 1) == []          # half a reading must not count as a good one, nor fault yet
    assert run(w, [rd(1.0 + i * 0.5, temp=None, hum=50.0) for i in range(700)])[-1].fault == "dht_dead"


def test_frozen_ldr_is_flagged_but_a_jittering_one_is_not():
    w = Watchdog()
    assert run(w, [rd(i * 0.5, ldr=1500 + (i % 3)) for i in range(4000)]) == []     # jitters of 2 counts for 2000 s
    w = Watchdog()
    ev = run(w, [rd(i * 0.5, ldr=1500) for i in range(1300)])                       # 650 s exactly constant
    assert [e.fault for e in ev] == ["ldr_flat"]


def test_constant_dark_or_bright_ldr_is_not_flat():
    for v in (0, 3, 4095):
        w = Watchdog()
        assert run(w, [rd(i * 0.5, ldr=v) for i in range(4000)]) == [], v            # ambiguous (night/day), left alone


def test_ldr_railed_needs_twelve_hours():
    w = Watchdog(HealthConfig(gap_s=1e9))        # the test samples once a minute
    assert run(w, [rd(t, ldr=4095) for t in range(0, 11 * 3600, 60)]) == []
    ev = run(w, [rd(t, ldr=4095) for t in range(11 * 3600, 13 * 3600, 60)])
    assert [e.fault for e in ev] == ["ldr_railed"]
    assert [e.kind for e in run(w, [rd(13 * 3600 + 60, ldr=2000)])] == ["sensor_fault_cleared"]


def test_stream_gap_is_reported_once_with_its_length():
    w = Watchdog()
    run(w, [rd(i * 0.5) for i in range(20)])
    ev = run(w, [rd(100.0)])
    assert len(ev) == 1 and ev[0].kind == "stream_gap" and "90" in ev[0].detail
    assert run(w, [rd(100.5)]) == []


def test_healthy_simulated_rooms_raise_nothing(stream):
    assert run(Watchdog(), stream) == []
    for profile in ("dht11", "dark_corridor"):                      # the DHT11 profile has nulls and 1-degree steps
        wc, sc = profile_configs(profile)
        assert run(Watchdog(), list(SimulatorSource(World(2, wc), 24 * 3600, sc))) == [], profile


def test_pipeline_reports_dead_dht_and_logs_it_without_changing_decisions(small_cfg, new_consensus, stream):
    dead = [replace(r, temp=None, hum=None) if 5 * 3600 <= r.t - stream[0].t < 5.5 * 3600 else r for r in stream]
    led = Ledger()
    a = Pipeline(small_cfg, consensus=new_consensus, ledger=led)
    a.start_calibration()
    out = feed(a, dead)
    kinds = [(e.kind, e.sensor) for o in out for e in o.events if e.kind.startswith("sensor_fault")]
    assert ("sensor_fault", "dht") in kinds and ("sensor_fault_cleared", "dht") in kinds
    assert [r["event"] for r in led.records if r["event"].startswith("sensor_fault")] == ["sensor_fault", "sensor_fault_cleared"]
    assert led.verify()[0]
    # the watchdog only reports: the same input with the watchdog's limits made unreachable gives identical decisions
    b = Pipeline(small_cfg, consensus=new_consensus, health=HealthConfig(dht_dead_s=1e12, ldr_flat_s=1e12, ldr_rail_s=1e12, gap_s=1e12))
    b.start_calibration()
    assert key(feed(b, dead)) == key(out)


def test_runner_writes_a_log_and_a_summary(tmp_path, stream):
    log = tmp_path / "logs" / "run.log"
    lines = []
    assert _main(["--log", str(log), "--quiet"], source=stream[:7200], echo=lines.append) == 0
    text = log.read_text()
    assert "calibrating" in text and "done:" in text and "input:" not in text      # a plain list has no source stats
    assert all(len(l) >= 19 and l[4] == "-" for l in text.splitlines() if l)       # every line time-stamped