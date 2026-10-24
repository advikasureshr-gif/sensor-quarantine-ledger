"""
preprocess.py - raw Reading -> CleanReading (the "Preprocessing / filtering" box in the flow).

Incremental and stateless between runs: feed readings in time order, get one CleanReading
per Reading. Works identically on simulator, replay and live serial input.

What it does
------------
DHT   null / impossible values -> hold the last good value for up to `dht_hold_max_s`,
      then report missing (None). `dht_age_s` says how stale the value is.
PIR   keeps the raw level, and adds `pir_event` = a rising edge at least
      `pir_retrigger_lockout_s` after the previous event, plus `pir_high_s`, how long the
      output has been continuously high.
Door  debounced state plus `door_event` (+1 opened, -1 closed, 0 none), reported at the
      moment the new state has been confirmed (`door_debounce_s` after it first appeared).
LDR   raw ADC value plus an exponentially smoothed copy.
Gaps  a gap longer than `gap_reset_s` resets all filters; no events are produced by the
      first reading after a gap.

What it deliberately does NOT do
--------------------------------
* It never removes "outliers" that are merely unusual. A sudden +8 C reading is a possible
  attack, so only physically impossible values are rejected. Judging anomalies belongs to
  the detector, and a cleaner that hides attacks would defeat the project.
* It never sees or forwards ground-truth labels: CleanReading has no label fields, so
  nothing downstream can peek at them even if labelled readings are fed in.

Note for later stages: a PIR stuck HIGH produces no rising edges, so `pir_event` counts it
as silence. Feature extraction must therefore also use the PIR level (duty / `pir_high_s`),
not only the event count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .config import DEFAULT_CONFIG, PreprocessConfig
from .contract import Reading


@dataclass(frozen=True, slots=True)
class CleanReading:
    t: float
    temp: Optional[float]          # deg C, held across short DHT failures; None if unavailable
    hum: Optional[float]           # % RH, same
    dht_age_s: Optional[float]     # seconds since the last good DHT read (0.0 = fresh); None = never
    pir: int                       # raw PIR level, 0/1
    pir_event: bool                # a new motion event (rising edge, after lockout)
    pir_high_s: float              # seconds the PIR has been continuously high (0.0 when low)
    door: int                      # debounced door state, 1 = open
    door_event: int                # +1 just opened, -1 just closed, 0 otherwise
    ldr_raw: int                   # raw ADC counts
    ldr: float                     # smoothed ADC counts


@dataclass
class PreprocessStats:
    readings: int = 0
    dht_missing: int = 0           # no usable DHT value at all (hold expired / never had one)
    dht_held: int = 0              # a held value was substituted
    dht_out_of_range: int = 0      # impossible values rejected
    pir_events: int = 0
    pir_lockout_suppressed: int = 0
    door_events: int = 0
    door_flickers_rejected: int = 0
    gap_resets: int = 0

    def summary(self) -> str:
        return (f"{self.readings} readings | DHT held {self.dht_held}, missing {self.dht_missing}, "
                f"out-of-range {self.dht_out_of_range} | PIR events {self.pir_events} "
                f"(lockout suppressed {self.pir_lockout_suppressed}) | door events {self.door_events} "
                f"(flickers rejected {self.door_flickers_rejected}) | gap resets {self.gap_resets}")


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig = DEFAULT_CONFIG.preprocess) -> None:
        self.cfg = cfg
        self.stats = PreprocessStats()
        self._last_t: Optional[float] = None
        self._reset_filters()

    # ---- state ---------------------------------------------------------------
    def _reset_filters(self) -> None:
        self._temp: Optional[float] = None
        self._hum: Optional[float] = None
        self._dht_good_t: Optional[float] = None
        self._pir_prev: Optional[int] = None
        self._pir_last_event_t = -math.inf
        self._pir_high_since: Optional[float] = None
        self._door: Optional[int] = None
        self._door_cand_t: Optional[float] = None     # when the not-yet-confirmed new state first appeared
        self._ldr_s: Optional[float] = None

    def reset(self) -> None:
        """Forget all filter state (not the statistics)."""
        self._last_t = None
        self._reset_filters()

    # ---- per-channel steps -----------------------------------------------------
    def _dht(self, r: Reading) -> tuple[Optional[float], Optional[float], Optional[float]]:
        c = self.cfg
        t, h = r.temp, r.hum
        bad = False
        for v, (lo, hi) in ((t, c.temp_valid_range_c), (h, c.hum_valid_range)):
            if v is not None and not lo <= v <= hi:
                self.stats.dht_out_of_range += 1
                bad = True
        if t is not None and h is not None and not bad:
            self._temp, self._hum, self._dht_good_t = t, h, r.t
            return t, h, 0.0
        # failed or impossible read: hold the last good value for a while
        if self._dht_good_t is not None:
            age = r.t - self._dht_good_t
            if age <= c.dht_hold_max_s:
                self.stats.dht_held += 1
                return self._temp, self._hum, age
            self.stats.dht_missing += 1
            return None, None, age
        self.stats.dht_missing += 1
        return None, None, None

    def _pir(self, r: Reading) -> tuple[bool, float]:
        event = False
        if self._pir_prev == 0 and r.pir == 1:
            if r.t - self._pir_last_event_t >= self.cfg.pir_retrigger_lockout_s:
                event = True
                self._pir_last_event_t = r.t
                self.stats.pir_events += 1
            else:
                self.stats.pir_lockout_suppressed += 1
        if r.pir == 1:
            if self._pir_high_since is None:
                self._pir_high_since = r.t
            high_s = r.t - self._pir_high_since
        else:
            self._pir_high_since = None
            high_s = 0.0
        self._pir_prev = r.pir
        return event, high_s

    def _door_step(self, r: Reading) -> tuple[int, int]:
        if self._door is None:                       # first reading after start/reset: adopt, no event
            self._door = r.door
            return self._door, 0
        if r.door == self._door:
            if self._door_cand_t is not None:        # a candidate change flickered back
                self.stats.door_flickers_rejected += 1
                self._door_cand_t = None
            return self._door, 0
        if self._door_cand_t is None:
            self._door_cand_t = r.t
        if r.t - self._door_cand_t >= self.cfg.door_debounce_s - 1e-9:
            self._door = r.door
            self._door_cand_t = None
            self.stats.door_events += 1
            return self._door, (1 if self._door == 1 else -1)
        return self._door, 0

    def _ldr(self, r: Reading) -> float:
        if self._ldr_s is None:
            self._ldr_s = float(r.ldr)
        else:
            self._ldr_s += self.cfg.ldr_ema_alpha * (r.ldr - self._ldr_s)
        return self._ldr_s

    # ---- public ------------------------------------------------------------------
    def process(self, r: Reading) -> CleanReading:
        if self._last_t is not None:
            if r.t < self._last_t:
                raise ValueError(f"time went backwards ({self._last_t} -> {r.t}); feed readings through a Source")
            if r.t - self._last_t > self.cfg.gap_reset_s:
                self._reset_filters()
                self.stats.gap_resets += 1
        self._last_t = r.t
        self.stats.readings += 1

        temp, hum, age = self._dht(r)
        pir_event, pir_high_s = self._pir(r)
        door, door_event = self._door_step(r)
        ldr_s = self._ldr(r)
        return CleanReading(t=r.t, temp=temp, hum=hum, dht_age_s=age, pir=r.pir, pir_event=pir_event,
                            pir_high_s=pir_high_s, door=door, door_event=door_event, ldr_raw=r.ldr, ldr=ldr_s)


def clean(readings: Iterable[Reading], cfg: PreprocessConfig = DEFAULT_CONFIG.preprocess) -> Iterator[CleanReading]:
    """Convenience wrapper: a stream of Readings -> a stream of CleanReadings."""
    pre = Preprocessor(cfg)
    for r in readings:
        yield pre.process(r)