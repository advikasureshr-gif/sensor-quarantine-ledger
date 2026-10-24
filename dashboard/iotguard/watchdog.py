"""
watchdog.py - is the hardware still delivering sensible data?  (reports only; it never changes a decision)

The detector judges whether the ROOM is behaving normally. This module judges something different: whether a
sensor or the data stream has stopped working, which looks like "nothing happening" to the detector and could
otherwise go unnoticed for days. It looks at the raw readings only and has no access to the model or the gate,
so adding it cannot change any detection result.

It reports four things, each when it STARTS and (except for the one-off gap) when it ENDS:

  stream_gap        no reading for longer than `gap_s` (one-off, reported when data resumes, with the length of the gap)
  dht_dead          no valid temperature/humidity for `dht_dead_s` (DHT11 not wired, wrong pin, or failing)
  ldr_flat          the light value has been EXACTLY the same, away from both ends of the scale, for `ldr_flat_s`.
                    An ADC that is really connected jitters by a count or two, so a perfectly constant value means the
                    reading is frozen, not that the room is steady.
  ldr_railed        the light value has sat at the very bottom or top of the scale for `ldr_rail_s` (default 12 h:
                    a short stretch is just a dark night or a bright day, so only a very long one is suspicious;
                    it can mean the LDR is disconnected or shorted)

What it deliberately does NOT try to detect:
  * A PIR or door that never changes. An empty room and a closed door look exactly like that, so there is no honest
    way to call it a fault. A PIR stuck HIGH is already caught by the consensus engine.
  * A DHT that reports plausible but wrong values. That is what the anomaly detector is for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contract import Reading


@dataclass(frozen=True)
class HealthConfig:
    gap_s: float = 10.0                 # readings arrive every 0.5 s; the preprocessor itself resets filters after 30 s
    dht_dead_s: float = 300.0
    ldr_flat_s: float = 600.0
    ldr_rail_lo: int = 5                # raw ADC counts at or below this count as "bottom of the scale"
    ldr_rail_hi: int = 4090
    ldr_rail_s: float = 12 * 3600.0


@dataclass(frozen=True)
class HealthEvent:
    kind: str                           # stream_gap | sensor_fault | sensor_fault_cleared
    t: float
    sensor: Optional[str]               # dht | ldr, or None for the stream
    fault: str                          # stream_gap | dht_dead | ldr_flat | ldr_railed
    detail: str


class Watchdog:
    def __init__(self, cfg: HealthConfig = HealthConfig()) -> None:
        self.cfg = cfg
        self.active: dict[str, float] = {}          # fault name -> time it started
        self._last_t: Optional[float] = None
        self._dht_bad_since: Optional[float] = None
        self._ldr_val: Optional[int] = None
        self._ldr_same_since: Optional[float] = None
        self._rail_since: Optional[float] = None
        self.history: list[HealthEvent] = []

    # ---- helpers ------------------------------------------------------------------------------------------
    def _emit(self, out: list, ev: HealthEvent) -> None:
        out.append(ev)
        self.history.append(ev)

    def _set(self, out: list, fault: str, sensor: str, on: bool, t: float, detail: str) -> None:
        if on and fault not in self.active:
            self.active[fault] = t
            self._emit(out, HealthEvent("sensor_fault", t, sensor, fault, detail))
        elif not on and fault in self.active:
            began = self.active.pop(fault)
            self._emit(out, HealthEvent("sensor_fault_cleared", t, sensor, fault, f"{fault} ended after {(t - began) / 60:.1f} min"))

    # ---- one reading ------------------------------------------------------------------------------------------
    def observe(self, r: Reading) -> list[HealthEvent]:
        c, t, out = self.cfg, r.t, []

        if self._last_t is not None and t - self._last_t > c.gap_s:
            self._emit(out, HealthEvent("stream_gap", t, None, "stream_gap",
                                        f"no data for {t - self._last_t:.1f} s (last reading at t={self._last_t:.1f} s)"))
            # nothing can be said about the time in between, so start every timer afresh
            self._dht_bad_since = self._ldr_same_since = self._rail_since = None
            self._ldr_val = None
        self._last_t = t

        # DHT: valid temperature AND humidity
        if r.temp is not None and r.hum is not None:
            self._dht_bad_since = None
            self._set(out, "dht_dead", "dht", False, t, "")
        else:
            if self._dht_bad_since is None:
                self._dht_bad_since = t
            bad = t - self._dht_bad_since
            self._set(out, "dht_dead", "dht", bad >= c.dht_dead_s, t,
                      f"no valid DHT reading for {bad / 60:.1f} min (check wiring, the data pin and its pull-up)")

        # LDR: frozen value
        if r.ldr == self._ldr_val:
            same = t - (self._ldr_same_since if self._ldr_same_since is not None else t)
        else:
            self._ldr_val, self._ldr_same_since, same = r.ldr, t, 0.0
        mid = c.ldr_rail_lo < r.ldr < c.ldr_rail_hi
        self._set(out, "ldr_flat", "ldr", mid and same >= c.ldr_flat_s, t,
                  f"light value stuck at exactly {r.ldr} for {same / 60:.1f} min (a connected ADC jitters; check the LDR wiring)")

        # LDR: pinned at an end of the scale for a very long time
        if r.ldr <= c.ldr_rail_lo or r.ldr >= c.ldr_rail_hi:
            if self._rail_since is None:
                self._rail_since = t
            railed = t - self._rail_since
            self._set(out, "ldr_railed", "ldr", railed >= c.ldr_rail_s, t,
                      f"light value at the end of the scale ({r.ldr}) for {railed / 3600:.1f} h (LDR disconnected or shorted?)")
        else:
            self._rail_since = None
            self._set(out, "ldr_railed", "ldr", False, t, "")
        return out

    def active_faults(self) -> dict[str, float]:
        return dict(self.active)