"""
sim_sensors.py - what the four sensors REPORT about the room, plus attacks and labels.

    sim_world.py    true state of the room
    sim_sensors.py  imperfect sensors -> Reading, attack injection, labels   <- this file
    (scenarios.py   named presets that combine the two; next file)

Pipeline per tick:   WorldState --SensorBank--> genuine values --AttackInjector--> Reading

Sensor models (each mimics the real part, including its quirks)
---------------------------------------------------------------
DHT   Refreshes every 2 s, so values repeat across ~4 lines (like the real firmware).
      Slow thermal lag, per-device bias, noise, quantization (0.1 for DHT22; use
      `SensorConfig.dht11()` for 1.0 steps), and read failures in bursts -> null.
PIR   HC-SR501 behavior: output stays high for a hold time after the last motion,
      then ignores motion for a short blocking time. Misses some motion, and fires
      occasionally with nobody there (false triggers).
Reed  Contact chatter while the door is closing (door "ajar"), and rare 1-sample
      bounce delays on transitions.
LDR   Saturating light response (so a corridor light adds less in a bright room),
      per-device gain, ADC noise, 12-bit integer output.

Attacks
-------
An attack manipulates ONE reported channel while the other sensors keep telling the
truth. That disagreement is exactly what cross-sensor consensus must notice.
    offset   add `magnitude` to the channel (temp C, hum %, ldr counts), scaled by intensity
    force    with probability = intensity, report `magnitude` instead of the truth
             (e.g. PIR stuck high = force pir to 1)
    replay   report what this channel genuinely said `lag_s` seconds earlier
`ramp_s` makes intensity grow linearly 0 -> 1 after `start_s` (gradual poisoning);
`ramp_s = 0` is an immediate (sudden) attack.

Ground truth: label = "attack" (with attack_kind = tag) from `start_s` while the
attack's intensity > 0, else "legit_change" while the world is in a legitimate-change
window, else "normal". Note that during a slow ramp the label says "attack" from the
very start even though the effect is initially tiny; the evaluator should account for
that when measuring detection latency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterator, Optional, Sequence

import numpy as np

from .contract import ADC_MAX, Reading
from .sim_world import World, WorldState
from .sources import Source

CHANNELS = ("temp", "hum", "pir", "door", "ldr")
ATTACK_KINDS = ("offset", "force", "replay")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorConfig:
    # --- DHT22 -----------------------------------------------------------------
    dht_period_s: float = 2.0          # DHT22 cannot refresh faster than 0.5 Hz
    dht_temp_res: float = 0.1          # output step (deg C)
    dht_hum_res: float = 0.1           # output step (% RH)
    dht_temp_noise: float = 0.15       # per-read noise sd (deg C)
    dht_hum_noise: float = 0.6         # per-read noise sd (% RH)
    dht_temp_bias_sd: float = 0.3      # per-device constant offset, drawn once
    dht_hum_bias_sd: float = 2.0
    dht_lag_s: float = 8.0             # first-order response time of the sensor housing
    dht_fail_p: float = 0.01           # chance a read fails
    dht_fail_continue_p: float = 0.5   # chance a failure is followed by another failure

    # --- PIR (HC-SR501) ----------------------------------------------------------
    pir_hold_s: float = 3.0            # output stays high this long after the last motion
    pir_block_s: float = 2.5           # ignores motion for this long after going low
    pir_detect_p: float = 0.8          # per-tick chance of noticing physical motion
    pir_false_per_h: float = 0.5       # spurious triggers with nobody there

    # --- reed switch ---------------------------------------------------------------
    door_chatter_p: float = 0.15       # chance a closing door chatters for 1-3 samples
    door_bounce_p: float = 0.02        # chance a transition is reported one sample late

    # --- LDR + 12-bit ADC ------------------------------------------------------------
    ldr_dark: float = 60.0             # ADC counts in a dark room
    ldr_span: float = 3600.0           # ADC counts added going from dark to saturated
    ldr_light_scale: float = 1.2       # saturation: adc = dark + span*(1 - exp(-light/scale))
    ldr_noise: float = 6.0             # ADC noise sd in counts (after firmware averaging)
    ldr_gain_sd: float = 0.05          # per-device multiplicative gain error
    ldr_invert: bool = False           # True if the module reads LOWER when brighter

    def __post_init__(self) -> None:
        if self.dht_period_s <= 0 or self.dht_lag_s <= 0 or self.ldr_light_scale <= 0:
            raise ValueError("dht_period_s, dht_lag_s and ldr_light_scale must be > 0")
        for name in ("dht_fail_p", "dht_fail_continue_p", "pir_detect_p", "door_chatter_p", "door_bounce_p"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @classmethod
    def dht11(cls, **overrides) -> "SensorConfig":
        """Coarser, noisier DHT11 variant (1.0 steps, +-2 C / +-5 % RH class accuracy)."""
        base = cls(dht_temp_res=1.0, dht_hum_res=1.0, dht_temp_noise=0.4, dht_hum_noise=1.5,
                   dht_temp_bias_sd=1.0, dht_hum_bias_sd=3.0)
        return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# buffered random numbers (scalar numpy draws are slow; this keeps runs fast and deterministic)
# ---------------------------------------------------------------------------
class _Stream:
    def __init__(self, rng: np.random.Generator, size: int = 8192) -> None:
        self._rng, self._size = rng, size
        self._u: list[float] = []
        self._n: list[float] = []
        self._iu = self._in = 0

    def u(self) -> float:
        if self._iu >= len(self._u):
            self._u, self._iu = self._rng.random(self._size).tolist(), 0
        v = self._u[self._iu]
        self._iu += 1
        return v

    def n(self) -> float:
        if self._in >= len(self._n):
            self._n, self._in = self._rng.standard_normal(self._size).tolist(), 0
        v = self._n[self._in]
        self._in += 1
        return v


def _quant(v: float, res: float) -> float:
    return round(round(v / res) * res, 6)


# ---------------------------------------------------------------------------
# individual sensor models
# ---------------------------------------------------------------------------
class _Dht:
    def __init__(self, cfg: SensorConfig, rs: _Stream, dev: np.random.Generator, dt: float) -> None:
        self.cfg, self.rs = cfg, rs
        self.every = max(1, int(round(cfg.dht_period_s / dt)))
        self.alpha = 1.0 - math.exp(-dt / cfg.dht_lag_s)
        self.t_bias = float(dev.normal(0.0, cfg.dht_temp_bias_sd))
        self.h_bias = float(dev.normal(0.0, cfg.dht_hum_bias_sd))
        self.k = 0
        self.st: Optional[float] = None     # sensor-internal (lagged) temperature
        self.sh: Optional[float] = None
        self.failing = False
        self.val: tuple[Optional[float], Optional[float]] = (None, None)

    def step(self, true_t: float, true_h: float) -> tuple[Optional[float], Optional[float]]:
        c = self.cfg
        if self.st is None:
            self.st, self.sh = true_t, true_h
        else:
            self.st += (true_t - self.st) * self.alpha
            self.sh += (true_h - self.sh) * self.alpha
        if self.k % self.every == 0:                       # a new read happens now
            p = c.dht_fail_continue_p if self.failing else c.dht_fail_p
            self.failing = self.rs.u() < p
            if self.failing:
                self.val = (None, None)                    # firmware prints null until next good read
            else:
                t = _quant(self.st + self.t_bias + c.dht_temp_noise * self.rs.n(), c.dht_temp_res)
                h = _quant(self.sh + self.h_bias + c.dht_hum_noise * self.rs.n(), c.dht_hum_res)
                self.val = (t, min(100.0, max(0.0, h)))
        self.k += 1
        return self.val


class _Pir:
    def __init__(self, cfg: SensorConfig, rs: _Stream, dt: float) -> None:
        self.cfg, self.rs, self.dt = cfg, rs, dt
        self.false_p = cfg.pir_false_per_h * dt / 3600.0
        self.out = 0
        self.high_until = 0.0
        self.blocked_until = 0.0

    def step(self, t: float, motion: bool) -> int:
        c = self.cfg
        seen = (motion and self.rs.u() < c.pir_detect_p) or (self.rs.u() < self.false_p)
        if self.out:
            if seen:
                self.high_until = t + c.pir_hold_s          # retrigger extends the high period
            if t >= self.high_until:
                self.out = 0
                self.blocked_until = t + c.pir_block_s
        elif seen and t >= self.blocked_until:
            self.out = 1
            self.high_until = t + c.pir_hold_s
        return self.out


class _Reed:
    def __init__(self, cfg: SensorConfig, rs: _Stream) -> None:
        self.cfg, self.rs = cfg, rs
        self.prev: Optional[int] = None
        self.rep = 0
        self.chatter = 0
        self.delay = 0

    def step(self, is_open: bool) -> int:
        o = 1 if is_open else 0
        if self.prev is None:
            self.rep = o
        elif o != self.prev:
            if o == 0 and self.rs.u() < self.cfg.door_chatter_p:
                self.chatter = 1 + int(self.rs.u() * 3)       # door ajar: 1-3 flickering samples
            elif self.rs.u() < self.cfg.door_bounce_p:
                self.delay = 1                                # transition reported one sample late
        self.prev = o
        if self.chatter > 0:
            self.chatter -= 1
            out = 1 if self.rs.u() < 0.5 else 0
        elif self.delay > 0:
            self.delay -= 1
            out = self.rep
        else:
            out = o
        self.rep = out
        return out


class _Ldr:
    def __init__(self, cfg: SensorConfig, rs: _Stream, dev: np.random.Generator) -> None:
        self.cfg, self.rs = cfg, rs
        self.gain = float(max(0.5, dev.normal(1.0, cfg.ldr_gain_sd)))

    def step(self, light: float) -> int:
        c = self.cfg
        x = c.ldr_dark + c.ldr_span * (1.0 - math.exp(-max(0.0, light) * self.gain / c.ldr_light_scale))
        x += c.ldr_noise * self.rs.n()
        v = int(round(min(ADC_MAX, max(0.0, x))))
        return ADC_MAX - v if c.ldr_invert else v


class SensorBank:
    """All four sensors, stateful. Call `read` once per tick, in order."""

    def __init__(self, cfg: SensorConfig, seed: int, dt: float) -> None:
        dev = np.random.default_rng([seed, 10, 0])            # per-device constants
        stream = lambda k: _Stream(np.random.default_rng([seed, 10, k]))
        self.dht = _Dht(cfg, stream(1), dev, dt)
        self.pir = _Pir(cfg, stream(2), dt)
        self.reed = _Reed(cfg, stream(3))
        self.ldr = _Ldr(cfg, stream(4), dev)

    def read(self, s: WorldState) -> dict:
        temp, hum = self.dht.step(s.temp, s.hum)
        return {"temp": temp, "hum": hum,
                "pir": self.pir.step(s.t, s.motion),
                "door": self.reed.step(s.door_open),
                "ldr": self.ldr.step(s.light)}


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Attack:
    kind: str                    # "offset" | "force" | "replay"
    channel: str                 # "temp" | "hum" | "pir" | "door" | "ldr"
    start_s: float
    end_s: float = math.inf
    magnitude: float = 0.0       # offset: amount added; force: value reported
    ramp_s: float = 0.0          # 0 = sudden; >0 = intensity grows 0 -> 1 over this long
    lag_s: float = 0.0           # replay only
    tag: str = "attack"          # becomes Reading.attack_kind
    peak: float = 1.0            # intensity reached at the end of the ramp (force/replay: fraction of
                                 # samples affected, e.g. 0.3 = PIR forced high on 30% of samples)

    def __post_init__(self) -> None:
        if self.kind not in ATTACK_KINDS:
            raise ValueError(f"kind must be one of {ATTACK_KINDS}, got {self.kind!r}")
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if not (0 <= self.start_s < self.end_s):
            raise ValueError("need 0 <= start_s < end_s")
        if self.ramp_s < 0 or not math.isfinite(self.magnitude):
            raise ValueError("ramp_s must be >= 0 and magnitude finite")
        if self.kind == "offset" and self.channel in ("pir", "door"):
            raise ValueError("offset attacks apply to temp, hum and ldr only; use kind='force' for pir/door")
        if self.kind == "force" and self.channel in ("pir", "door") and self.magnitude not in (0, 1):
            raise ValueError("forcing pir/door requires magnitude 0 or 1")
        if self.kind == "replay" and self.lag_s <= 0:
            raise ValueError("replay attacks need lag_s > 0")
        if not self.tag:
            raise ValueError("tag must be non-empty")
        if not 0.0 < self.peak <= 1.0:
            raise ValueError("peak must be in (0, 1]")

    def intensity(self, t: float) -> float:
        if t < self.start_s or t >= self.end_s:
            return 0.0
        frac = 1.0 if self.ramp_s <= 0 else min(1.0, (t - self.start_s) / self.ramp_s)
        return max(self.peak * frac, 1e-9)       # exactly at start: tiny but still "active"


class AttackInjector:
    def __init__(self, attacks: Sequence[Attack], dt: float, seed: int) -> None:
        self.attacks = tuple(attacks)
        self.rs = _Stream(np.random.default_rng([seed, 10, 5]))
        self._hist: dict[int, deque] = {
            i: deque(maxlen=max(1, int(round(a.lag_s / dt))))
            for i, a in enumerate(self.attacks) if a.kind == "replay"
        }

    @staticmethod
    def _fit(channel: str, v: float):
        if channel == "ldr":
            return int(round(min(ADC_MAX, max(0.0, v))))
        if channel == "hum":
            return round(min(100.0, max(0.0, v)), 1)
        if channel == "temp":
            return round(v, 1)
        return int(v)

    def apply(self, t: float, genuine: dict) -> tuple[dict, Optional[str]]:
        """Returns (reported values, attack_kind tag or None when no attack is active)."""
        out = dict(genuine)
        tags: list[str] = []
        for i, a in enumerate(self.attacks):
            lagged = None
            if a.kind == "replay":                       # always record, so history exists when the attack starts
                dq = self._hist[i]
                lagged = dq[0] if len(dq) == dq.maxlen else None
                dq.append(genuine[a.channel])
            inten = a.intensity(t)
            if inten <= 0.0:
                continue
            tags.append(a.tag)
            # offset scales its own size by intensity; force/replay instead take effect on a
            # fraction `intensity` of the ticks (applying both would make the ramp quadratic).
            if a.kind != "offset" and inten < 1.0 and self.rs.u() >= inten:
                continue
            ch = a.channel
            if a.kind == "offset":
                if out[ch] is not None:
                    out[ch] = self._fit(ch, out[ch] + a.magnitude * inten)
            elif a.kind == "force":
                out[ch] = self._fit(ch, a.magnitude)
            elif lagged is not None:                     # replay
                out[ch] = lagged
        return out, ("+".join(sorted(set(tags))) if tags else None)


# ---------------------------------------------------------------------------
# the source the pipeline will actually read
# ---------------------------------------------------------------------------
class SimulatorSource(Source):
    """
    World + sensors + attacks as a `Source`. Readings carry ground-truth labels;
    feed the pipeline through `sources.strip_labels` so it never sees them.
    Re-iterating regenerates the identical stream (same seed).
    """

    def __init__(self, world: World, duration_s: float, sensors: SensorConfig = SensorConfig(),
                 attacks: Sequence[Attack] = (), seed: Optional[int] = None, **kw) -> None:
        kw.setdefault("sample_period_s", world.cfg.sample_period_s)
        super().__init__(**kw)
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        self.world, self.duration_s = world, float(duration_s)
        self.sensors, self.attacks = sensors, tuple(attacks)
        self.seed = world.seed if seed is None else int(seed)

    def _raw(self) -> Iterator[Reading]:
        dt = self.world.cfg.sample_period_s
        bank = SensorBank(self.sensors, self.seed, dt)
        inj = AttackInjector(self.attacks, dt, self.seed)
        for st in self.world.states(self.duration_s):
            vals, tag = inj.apply(st.t, bank.read(st))
            if tag is not None:
                label, kind = "attack", tag
            elif st.legit_change is not None:
                label, kind = "legit_change", None
            else:
                label, kind = "normal", None
            yield Reading(t=round(st.t, 3), temp=vals["temp"], hum=vals["hum"], pir=vals["pir"],
                          door=vals["door"], ldr=vals["ldr"], label=label, attack_kind=kind)