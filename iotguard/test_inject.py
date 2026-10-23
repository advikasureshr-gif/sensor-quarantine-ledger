"""The attack injector: alters only what it is told to, only inside its window, and nothing downstream can tell."""
import pytest

from iotguard.contract import Reading
from iotguard.inject import Injection, InjectSource, quiet_stretches


def stream(hours=3.0, step=60.0, **kw):
    n = int(hours * 3600 / step)
    base = dict(temp=25.0, hum=50.0, pir=0, door=0, ldr=1000)
    base.update(kw)
    return [Reading(t=i * step, **base) for i in range(n)]


def test_force_pir_only_inside_the_window():
    src = stream()
    out = list(InjectSource(src, [Injection("pir", start_h=1.0, hours=1.0)]))
    assert len(out) == len(src)
    for before, after in zip(src, out):
        inside = 1.0 <= before.t / 3600 < 2.0
        assert after.pir == (1 if inside else 0)
        assert (after.temp, after.hum, after.door, after.ldr) == (before.temp, before.hum, before.door, before.ldr)


def test_offset_ldr_is_clamped_and_dht_failure_stays_failed():
    src = [Reading(0.0, 25.0, 50.0, 0, 0, 4000), Reading(1.0, None, None, 0, 0, 4000)]
    out = list(InjectSource(src, [Injection("ldr", 0, 1, "offset", 500), Injection("temp", 0, 1, "offset", 5.0)]))
    assert out[0].ldr == 4095 and out[0].temp == 30.0
    assert out[1].temp is None and out[1].hum is None


def test_start_and_end_are_reported_once():
    seen = []
    list(InjectSource(stream(), [Injection("pir", 1.0, 1.0)], on_change=lambda ev, inj, t: seen.append((ev, round(t / 3600, 2)))))
    assert seen == [("start", 1.0), ("end", 2.0)]


def test_bad_injections_are_refused():
    with pytest.raises(ValueError):
        Injection("pir", 0, 1, "offset", 1)
    with pytest.raises(ValueError):
        Injection("pir", 0, 1, "force", 5)
    with pytest.raises(ValueError):
        Injection("gps", 0, 1)
    with pytest.raises(ValueError):
        Injection("pir", 0, 0)


def test_quiet_stretches_ignore_short_pir_blips_but_not_door_changes():
    rs = []
    for i in range(0, 7200, 10):                     # 2 hours at 10 s
        pir = 1 if i in (1000, 1010) else 0          # a 20 s blip... counts (>= 6 s)
        door = 1 if i >= 5000 else 0                 # a door change at 5000 s
        rs.append(Reading(float(i), 25.0, 50.0, pir, door, 1000))
    stretches, total = quiet_stretches(rs)
    edges = [round(s * 3600) for s, _ in stretches] + [round(e * 3600) for _, e in stretches]
    assert 1010 in edges and 5000 in edges
    assert abs(total - 7190 / 3600) < 1e-6
