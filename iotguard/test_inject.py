"""
inject.py - attack a REAL (recorded or live) stream in software, without touching the sensors.

    InjectSource(source, [Injection("pir", start_h=56, hours=4)])      wraps any iterable of Readings
    python -m iotguard.inject --suggest logs/room.jsonl --hours 4       find a quiet stretch to attack

The wrapped stream is identical to the original except inside the injection windows, where the chosen channel is
forced or shifted. Windows are measured in hours since the FIRST reading, on the source's own clock (so a recording,
a replay and a live run behave alike). Nothing downstream can tell an injected reading from a real one: the pipeline
sees only Readings, exactly as in a real attack on a sensor or on the data path.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Optional

from .contract import ADC_MAX, Reading

CHANNELS = ("temp", "hum", "pir", "door", "ldr")


@dataclass(frozen=True)
class Injection:
    channel: str                 # temp | hum | pir | door | ldr
    start_h: float               # hours after the first reading
    hours: float                 # how long the attack lasts
    kind: str = "force"          # force: report `value`;  offset: add `value` (temp, hum, ldr only)
    value: float = 1.0
    tag: str = "injected"

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if self.kind not in ("force", "offset"):
            raise ValueError("kind must be 'force' or 'offset'")
        if self.start_h < 0 or self.hours <= 0:
            raise ValueError("need start_h >= 0 and hours > 0")
        if self.channel in ("pir", "door"):
            if self.kind != "force" or self.value not in (0, 1):
                raise ValueError("pir and door can only be forced to 0 or 1")

    @property
    def end_h(self) -> float:
        return self.start_h + self.hours

    def active(self, hours_since_start: float) -> bool:
        return self.start_h <= hours_since_start < self.end_h

    def apply(self, r: Reading) -> Reading:
        c, v = self.channel, self.value
        if c in ("pir", "door"):
            return replace(r, **{c: int(v)})
        if c == "ldr":
            x = v if self.kind == "force" else r.ldr + v
            return replace(r, ldr=int(min(ADC_MAX, max(0, round(x)))))
        cur = getattr(r, c)
        if cur is None:                                   # a failed DHT read stays failed
            return r
        return replace(r, **{c: float(v if self.kind == "force" else cur + v)})


class InjectSource:
    """Iterate to get the attacked stream. `on_change(event, injection, t)` fires when a window starts or ends."""

    def __init__(self, source: Iterable[Reading], injections: Iterable[Injection],
                 on_change: Optional[Callable[[str, Injection, float], None]] = None) -> None:
        self.source, self.injections, self.on_change = source, tuple(injections), on_change
        self.n_altered = 0

    @property
    def stats(self):
        return getattr(self.source, "stats", None)

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if close:
            close()

    def __iter__(self) -> Iterator[Reading]:
        t0: Optional[float] = None
        live: set[int] = set()
        done: set[int] = set()
        for r in self.source:
            if t0 is None:
                t0 = r.t
            h = (r.t - t0) / 3600.0
            for i, inj in enumerate(self.injections):
                if inj.active(h) and i not in live:
                    live.add(i)
                    if self.on_change:
                        self.on_change("start", inj, r.t)
                elif i in live and h >= inj.end_h and i not in done:
                    live.discard(i); done.add(i)
                    if self.on_change:
                        self.on_change("end", inj, r.t)
            out = r
            for i in live:
                out = self.injections[i].apply(out)
            if out is not r:
                self.n_altered += 1
            yield out


# ---------------------------------------------------------------------------
# finding a good moment to attack a recording
# ---------------------------------------------------------------------------
def quiet_stretches(readings: Iterable[Reading], blip_s: float = 6.0) -> tuple[list[tuple[float, float]], float]:
    """
    Hours-since-start intervals with no real activity (no PIR high longer than `blip_s`, no door change), and the
    total length of the recording in hours. A room believed EMPTY is where a forced PIR is a clear contradiction.
    """
    t0 = None
    start_quiet: Optional[float] = None
    last_door: Optional[int] = None
    pir_since: Optional[float] = None
    stretches: list[tuple[float, float]] = []
    last_t = 0.0
    for r in readings:
        if t0 is None:
            t0, start_quiet = r.t, r.t
        last_t = r.t
        activity = False
        if last_door is not None and r.door != last_door:
            activity = True
        last_door = r.door
        if r.pir:
            if pir_since is None:
                pir_since = r.t
            if r.t - pir_since >= blip_s:
                activity = True
        else:
            pir_since = None
        if activity:
            if start_quiet is not None and r.t > start_quiet:
                stretches.append(((start_quiet - t0) / 3600.0, (r.t - t0) / 3600.0))
            start_quiet = r.t
    if t0 is not None and start_quiet is not None:
        stretches.append(((start_quiet - t0) / 3600.0, (last_t - t0) / 3600.0))
    return stretches, ((last_t - t0) / 3600.0 if t0 is not None else 0.0)


def suggest(path: str, hours: float = 4.0, min_start_h: float = 0.0, tail_h: float = 3.5, lead_h: float = 0.25):
    """Candidate attack start times (hours): inside a long quiet stretch, with `tail_h` of data left for recovery."""
    from .sources import ReplaySource
    stretches, total_h = quiet_stretches(ReplaySource(path))
    out = []
    for s, e in stretches:
        a = max(s + lead_h, min_start_h)
        if a + hours <= e and a + hours + tail_h <= total_h:
            out.append((a, e - s))
    return sorted(out, key=lambda x: -x[1]), total_h


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suggest", required=True, metavar="RECORDING.jsonl")
    ap.add_argument("--hours", type=float, default=4.0, help="attack length (quarantine needs at least ~3.2 h)")
    ap.add_argument("--min-start-h", type=float, default=0.0, help="not before this many hours (e.g. after calibration)")
    ap.add_argument("--tail-h", type=float, default=3.5, help="data wanted after the attack, for the recovery step")
    a = ap.parse_args(argv)
    cands, total = suggest(a.suggest, a.hours, a.min_start_h, a.tail_h)
    print(f"recording length: {total:.1f} h")
    if not cands:
        print(f"no quiet stretch of {a.hours:g} h (starting after hour {a.min_start_h:g}, leaving {a.tail_h:g} h afterwards) was found.")
        print("Record longer, or lower --hours / --min-start-h.")
        return 1
    print(f"attack candidates ({a.hours:g} h long), best first:   --attack-start-h <value>")
    for start, quiet in cands[:5]:
        print(f"  {start:7.2f}   (quiet stretch of {quiet:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
