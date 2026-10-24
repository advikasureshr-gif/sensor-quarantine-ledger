"""
features.py - cleaned readings -> one feature vector per time window.

    CleanReading stream --WindowExtractor--> WindowFeatures

The 7-element vector (order fixed by `FeatureConfig.feature_names`):
    temp_mean, hum_mean   mean over the window's usable DHT samples
    pir_count             motion events (rising edges, after lockout)
    pir_duty              fraction of samples with the PIR high (catches a PIR stuck HIGH,
                          which has no rising edges and so counts as zero events)
    door_count            confirmed door transitions (opens + closes)
    ldr_mean, ldr_std     mean / spread of the smoothed LDR

A `WindowFeatures` also carries the EVIDENCE the consensus engine needs, which does not fit in
a fixed vector: the timestamped door events and PIR events, the longest continuous PIR-high
stretch, door state at the window's ends, LDR extremes and the largest raw sample-to-sample LDR
step, and the cleaned samples themselves.

Windowing rules
---------------
* Window k covers [k*hop, k*hop + window) on the SOURCE clock (half-open), so windows line up
  across live, replay and simulator runs.
* A window is emitted once a reading at or past its end arrives (purely timestamp-driven).
  The final partial window of a stream is dropped.
* Windows with no readings are not emitted (a long gap does not flood the pipeline).
* `valid` is False when too few samples arrived (coverage below `min_window_coverage`) or the
  vector contains a NaN. Invalid windows must not be used for learning or scoring.
* If a window has too few DHT samples, temp_mean/hum_mean are carried forward from the last good
  window (`dht_carried=True`). With nothing to carry they are NaN and the window is invalid.
* Overlapping windows (hop < window) are supported.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, FeatureConfig
from .preprocess import CleanReading


@dataclass(frozen=True, slots=True, eq=False)
class WindowFeatures:
    index: int                                  # window number k; t_start = k * hop
    t_start: float
    t_end: float
    names: tuple[str, ...]                      # names of the entries of x
    x: np.ndarray                               # feature vector (read-only)
    n: int                                      # samples in the window
    coverage: float                             # n / expected samples, capped at 1
    valid: bool
    dht_fraction: float                         # share of samples with a usable DHT value
    dht_carried: bool                           # temp/hum were carried forward from an earlier window
    # --- evidence for consensus --------------------------------------------------------
    door_events: tuple[tuple[float, int], ...]  # (time, +1 opened / -1 closed)
    pir_event_times: tuple[float, ...]
    pir_high_max_s: float                       # longest continuous PIR-high duration seen in window
    door_start: int
    door_end: int
    door_open_frac: float
    ldr_first: float                            # smoothed LDR at the first / last sample
    ldr_last: float
    ldr_min: float
    ldr_max: float
    ldr_step_max: float                         # largest |raw[i] - raw[i-1]| in the window
    samples: tuple[CleanReading, ...]           # the cleaned readings (empty if keep_samples=False)

    def feature(self, name: str) -> float:
        return float(self.x[self.names.index(name)])

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.names, self.x)}


class WindowExtractor:
    def __init__(self, cfg: FeatureConfig = DEFAULT_CONFIG.features,
                 sample_period_s: float = DEFAULT_CONFIG.sampling.sample_period_s,
                 keep_samples: bool = True) -> None:
        if cfg.window_s <= 0 or cfg.hop_s <= 0 or cfg.hop_s > cfg.window_s:
            raise ValueError("need window_s > 0 and 0 < hop_s <= window_s")
        if sample_period_s <= 0:
            raise ValueError("sample_period_s must be > 0")
        self.cfg = cfg
        self.keep_samples = keep_samples
        self.expected = max(1, int(round(cfg.window_s / sample_period_s)))
        self.reset()

    def reset(self) -> None:
        self._buf: deque[CleanReading] = deque()
        self._next_k: Optional[int] = None
        self._last_temp: Optional[float] = None
        self._last_hum: Optional[float] = None
        self._last_t = -math.inf

    # ---- streaming interface ------------------------------------------------------------
    def push(self, c: CleanReading) -> list[WindowFeatures]:
        """Feed one cleaned reading; returns the windows it completed (usually 0 or 1)."""
        if c.t < self._last_t:
            raise ValueError(f"time went backwards ({self._last_t} -> {c.t})")
        self._last_t = c.t
        W, hop = self.cfg.window_s, self.cfg.hop_s
        out: list[WindowFeatures] = []
        first_k = lambda t: max(0, math.floor((t - W) / hop + 1e-9) + 1)   # first window containing t
        if self._next_k is None:
            self._next_k = first_k(c.t)
        while c.t >= self._next_k * hop + W - 1e-9:
            w = self._finalize(self._next_k)
            if w is not None:
                out.append(w)
            self._next_k += 1
            if not self._buf and first_k(c.t) > self._next_k:      # after a gap: skip empty windows
                self._next_k = first_k(c.t)
        self._buf.append(c)
        return out

    def extract(self, stream: Iterable[CleanReading]) -> Iterator[WindowFeatures]:
        for c in stream:
            yield from self.push(c)

    # ---- one window ------------------------------------------------------------------------
    def _finalize(self, k: int) -> Optional[WindowFeatures]:
        cfg = self.cfg
        s, e = k * cfg.hop_s, k * cfg.hop_s + cfg.window_s
        samples = [c for c in self._buf if c.t >= s]
        cut = (k + 1) * cfg.hop_s
        while self._buf and self._buf[0].t < cut:
            self._buf.popleft()
        if not samples:
            return None
        n = len(samples)
        coverage = min(1.0, n / self.expected)

        # DHT
        temps = [c.temp for c in samples if c.temp is not None and c.hum is not None]
        hums = [c.hum for c in samples if c.temp is not None and c.hum is not None]
        dht_fraction = len(temps) / n
        carried = False
        if temps and dht_fraction >= cfg.min_dht_fraction:
            temp_mean, hum_mean = float(np.mean(temps)), float(np.mean(hums))
            self._last_temp, self._last_hum = temp_mean, hum_mean
        elif self._last_temp is not None:
            temp_mean, hum_mean, carried = self._last_temp, self._last_hum, True
        else:
            temp_mean = hum_mean = math.nan

        pir_times = tuple(c.t for c in samples if c.pir_event)
        door_events = tuple((c.t, c.door_event) for c in samples if c.door_event)
        smoothed = np.fromiter((c.ldr for c in samples), dtype=float, count=n)
        raw = np.fromiter((c.ldr_raw for c in samples), dtype=float, count=n)

        values = {
            "temp_mean": temp_mean, "hum_mean": hum_mean,
            "pir_count": float(len(pir_times)),
            "pir_duty": sum(c.pir for c in samples) / n,
            "door_count": float(len(door_events)),
            "ldr_mean": float(smoothed.mean()), "ldr_std": float(smoothed.std()),
        }
        x = np.array([values[name] for name in cfg.feature_names], dtype=float)   # KeyError = bad config
        x.flags.writeable = False

        return WindowFeatures(
            index=k, t_start=float(s), t_end=float(e), names=cfg.feature_names, x=x, n=n, coverage=coverage,
            valid=bool(coverage >= cfg.min_window_coverage and np.isfinite(x).all()),
            dht_fraction=dht_fraction, dht_carried=carried,
            door_events=door_events, pir_event_times=pir_times,
            pir_high_max_s=max(c.pir_high_s for c in samples),
            door_start=samples[0].door, door_end=samples[-1].door,
            door_open_frac=sum(c.door for c in samples) / n,
            ldr_first=float(smoothed[0]), ldr_last=float(smoothed[-1]),
            ldr_min=float(smoothed.min()), ldr_max=float(smoothed.max()),
            ldr_step_max=float(np.abs(np.diff(raw)).max()) if n > 1 else 0.0,
            samples=tuple(samples) if self.keep_samples else (),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def extract_windows(stream: Iterable[CleanReading], cfg: FeatureConfig = DEFAULT_CONFIG.features,
                    sample_period_s: float = DEFAULT_CONFIG.sampling.sample_period_s,
                    keep_samples: bool = True) -> Iterator[WindowFeatures]:
    """One-call convenience: CleanReadings -> WindowFeatures."""
    return WindowExtractor(cfg, sample_period_s, keep_samples).extract(stream)


def feature_matrix(windows: Sequence[WindowFeatures], valid_only: bool = True) -> np.ndarray:
    """Stack window vectors into an (n_windows, n_features) array."""
    rows = [w.x for w in windows if w.valid or not valid_only]
    return np.vstack(rows) if rows else np.empty((0, len(DEFAULT_CONFIG.features.feature_names)))