"""
sim_world.py - the TRUE state of the simulated room, before any sensor sees it.

Layering (this file is layer 1-3 of the simulator):

    sim_world.py    what is physically happening in the room        <- this file
    sim_sensors.py  what the DHT / PIR / reed / LDR *report* about it,
                    plus attack injectors and ground-truth labels   <- next file

The world yields one `WorldState` per tick (default 0.5 s, matching the ESP32).
It knows nothing about sensors, ADC counts, or attacks.

Physical couplings built in (the relationships the consensus engine will rely on)
-------------------------------------------------------------------------------
* Door opens  -> a person is passing through -> PIR motion within ~1 s.
* Door opens  -> corridor light enters the room -> `light` steps up.
* Occupied + dark -> the occupant switches the lamp on -> `light` steps up.
  The lamp goes off just before they leave.
* Occupied    -> body heat and humidity; lamp -> a little heat.
* Door open   -> the room briefly exchanges air with the corridor.
* Temperature follows outdoor temperature through a slow (hours) first-order lag.
* Humidity falls as temperature rises (and relaxes slowly toward its target).

Routine (what "normal" looks like)
----------------------------------
Weekdays: arrive ~09:00, lunch break ~13:00 (45 min), 1-4 short breaks, leave ~18:00,
occasional visitors, rarely a door propped open for ventilation. Weekends: often empty,
sometimes a short visit. Every number jitters day to day. All of this is generated from
`seed`, so a run is exactly reproducible. Day k's schedule does not depend on how many
days you simulate.

Legitimate environmental changes
--------------------------------
`LegitChange` models permanent, benign shifts (blinds removed, AC installed, a new
routine, a busier room). They are NOT attacks: the ground-truth label during and shortly
after them is "legit_change", and the system is expected to ADAPT to them.

Units
-----
temp: deg C.  hum: % RH.  light: dimensionless illuminance at the LDR; 0 = dark room,
~1 = bright; the sensor layer maps it to ADC counts. t: seconds since start of run.
Simulator parameters live here (not in config.py) because they describe the fake world,
not the pipeline; that keeps pipeline tuning from ever touching the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG

DAY_S = 86400.0
CHANGE_KINDS = ("brighter", "warmer", "busier", "shifted")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldConfig:
    # --- time ---------------------------------------------------------------
    sample_period_s: float = DEFAULT_CONFIG.sampling.sample_period_s
    start_hour: float = 8.0            # time of day at t = 0
    start_day_of_week: int = 0         # 0 = Monday ... 6 = Sunday
    chunk_s: float = 3600.0            # internal batch size; no effect on results

    # --- occupant routine (hours are clock hours) ---------------------------
    arrive_mean_h: float = 9.0
    arrive_sd_h: float = 0.35
    leave_mean_h: float = 18.0
    leave_sd_h: float = 0.6
    lunch_prob: float = 0.8
    lunch_mean_h: float = 13.0
    lunch_sd_h: float = 0.3
    lunch_dur_mean_min: float = 45.0
    lunch_dur_sd_min: float = 12.0
    breaks_per_day: float = 2.5        # Poisson mean of short absences
    break_dur_mean_min: float = 6.0
    absent_prob: float = 0.05          # weekday nobody shows up
    weekend_presence_prob: float = 0.4
    visitors_per_day: float = 1.0      # Poisson mean, weekdays
    propped_door_prob: float = 0.08    # per weekday: door held open 5-30 min

    # --- door ----------------------------------------------------------------
    entry_open_s: tuple[float, float] = (4.0, 12.0)   # door open while entering
    exit_open_s: tuple[float, float] = (3.0, 8.0)     # door open while leaving

    # --- motion --------------------------------------------------------------
    burst_rate_per_h: float = 12.0     # desk activity bursts while occupied
    burst_mean_s: float = 15.0
    activity_hour_sigma: float = 0.4   # lognormal hour-to-hour variation in activity
    fidget_p: float = 0.002            # per-tick chance of a tiny movement when occupied
    visitor_burst_rate_per_h: float = 60.0

    # --- light ---------------------------------------------------------------
    sunrise_h: float = 6.0
    sunset_h: float = 18.5
    daylight_peak: float = 0.60        # daylight contribution at solar noon, clear sky
    sun_exponent: float = 1.3
    lamp_level: float = 0.50
    corridor_level: float = 0.12       # light entering while the door is open
    lamp_threshold_mean: float = 0.35  # occupant switches lamp on below this daylight level
    lamp_threshold_sd: float = 0.06
    cloud_min: float = 0.45
    cloud_tau_h: float = 1.0

    # --- thermal -------------------------------------------------------------
    out_mean_c: float = 28.0
    out_amp_c: float = 4.0
    out_peak_h: float = 15.0
    weather_sd_c: float = 1.5          # slow day-to-day outdoor wander
    weather_tau_h: float = 18.0
    tau_room_s: float = 3 * 3600.0     # room thermal time constant
    occ_gain_c: float = 1.2            # equilibrium warming from an occupant
    lamp_gain_c: float = 0.6           # equilibrium warming from the lamp
    corridor_temp_c: float = 27.0
    tau_door_s: float = 150.0          # air-exchange time constant while door is open
    temp_noise: float = 0.002          # process noise, deg C per sqrt(second)

    # --- humidity ------------------------------------------------------------
    rh_base: float = 60.0
    rh_occ_gain: float = 3.0
    rh_temp_coeff: float = 2.0         # % RH lost per deg C above t_ref
    t_ref_c: float = 28.0
    rh_weather_sd: float = 4.0
    tau_rh_s: float = 2 * 3600.0
    rh_corridor: float = 58.0
    rh_noise: float = 0.02             # % RH per sqrt(second)

    # --- abrupt natural light changes (a cloud edge, a reflection, a curtain moving) ---------------
    # Off by default. Each event multiplies the DAYLIGHT reaching the sensor by a factor drawn from
    # cloud_drop_range for a duration drawn from cloud_duration_range_s, switching instantly at both
    # ends. The occupant's lamp decisions ignore it. Used to test that benign lighting changes do
    # not look like attacks.
    cloud_events_per_h: float = 0.0
    cloud_drop_range: tuple[float, float] = (0.3, 0.7)
    cloud_duration_range_s: tuple[float, float] = (60.0, 600.0)

    def __post_init__(self) -> None:
        if self.sample_period_s <= 0:
            raise ValueError("sample_period_s must be > 0")
        if not 0.0 <= self.start_hour < 24.0:
            raise ValueError("start_hour must be in [0, 24)")
        if not 0 <= self.start_day_of_week <= 6:
            raise ValueError("start_day_of_week must be 0..6")
        if not self.sunrise_h < self.sunset_h:
            raise ValueError("sunrise_h must be < sunset_h")


@dataclass(frozen=True)
class LegitChange:
    """
    A permanent, benign change in the environment starting at `start_s` (sim seconds).

    kind        magnitude means
    ----------  ---------------------------------------------------------------
    brighter    fractional change in daylight reaching the sensor (+0.5 = 50% more;
                negative = dimmer, e.g. new curtains). Ramps in over ramp_s.
    warmer      deg C added to the outdoor temperature (negative = cooler, e.g. AC
                installed). Ramps in over ramp_s; indoor follows with the room's lag.
    busier      fractional change in activity and visitor rate (+1.0 = double;
                negative = quieter). Ramps in over ramp_s.
    shifted     hours added to the occupant's whole routine (+2 = arrives and leaves
                two hours later). Applies from the first day starting after start_s.

    Ground truth: the label is "legit_change" from start_s until start_s + ramp_s +
    settle_s, and "normal" afterwards (the new regime is the new normal).
    """
    kind: str
    start_s: float
    magnitude: float
    ramp_s: float = 600.0
    settle_s: float = 6 * 3600.0

    def __post_init__(self) -> None:
        if self.kind not in CHANGE_KINDS:
            raise ValueError(f"kind must be one of {CHANGE_KINDS}, got {self.kind!r}")
        if self.start_s < 0 or self.ramp_s < 0 or self.settle_s < 0:
            raise ValueError("start_s, ramp_s and settle_s must be >= 0")

    @property
    def label_end_s(self) -> float:
        return self.start_s + self.ramp_s + self.settle_s


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
class WorldState(NamedTuple):
    t: float                       # seconds since start of run
    tod_h: float                   # time of day in hours (debugging / plots only)
    temp: float                    # true air temperature, deg C
    hum: float                     # true relative humidity, %
    occupied: bool                 # someone is in the room
    motion: bool                   # physical movement in the PIR's field of view
    door_open: bool                # door physically open
    lamp_on: bool
    light: float                   # true illuminance at the LDR (see module docstring)
    legit_change: Optional[str]    # kind of an active legitimate change, else None


@dataclass
class DayEvents:
    """One day's generated event schedule, all in absolute clock seconds. For inspection/tests."""
    day: int
    sessions: list[tuple[float, float]] = field(default_factory=list)   # occupant door-in .. door-out
    presence: list[tuple[float, float]] = field(default_factory=list)
    door: list[tuple[float, float]] = field(default_factory=list)
    motion: list[tuple[float, float]] = field(default_factory=list)
    lamp: list[tuple[float, float, float]] = field(default_factory=list)  # (start, end, threshold)
    visitors: int = 0
    propped: int = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ou(rng: np.random.Generator, n: int, step_s: float, tau_s: float, sd: float) -> np.ndarray:
    """Ornstein-Uhlenbeck series (stationary sd `sd`). Prefix-stable in n for a fixed rng."""
    phi = math.exp(-step_s / tau_s)
    s = sd * math.sqrt(1.0 - phi * phi)
    z = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = sd * z[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + s * z[i]
    return x


def _idx(clock_s: float, start_s: float, dt: float) -> int:
    return int(math.ceil((clock_s - start_s) / dt))


def _paint(mask: np.ndarray, intervals, start_s: float, dt: float) -> None:
    n = mask.shape[0]
    for a, b in intervals:
        i0 = max(0, _idx(a, start_s, dt))
        i1 = min(n, _idx(b, start_s, dt))
        if i1 > i0:
            mask[i0:i1] = True


def _merge(iv: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(iv):
        if out and a < out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------
class World:
    def __init__(self, seed: int = 0, cfg: WorldConfig = WorldConfig(),
                 changes: Sequence[LegitChange] = ()) -> None:
        if seed < 0:
            raise ValueError("seed must be >= 0")
        self.seed = int(seed)
        self.cfg = cfg
        self.changes = tuple(changes)

    # ---- legitimate-change helpers -----------------------------------------
    @property
    def start_s(self) -> float:
        return self.cfg.start_hour * 3600.0

    def legit_windows(self) -> list[tuple[float, float, str]]:
        """(start, end, kind) sim-second windows labelled 'legit_change'. For the evaluator."""
        return [(c.start_s, c.label_end_s, c.kind) for c in self.changes]

    @staticmethod
    def _ramp(ch: LegitChange, t):
        if ch.ramp_s <= 0:
            return np.where(np.asarray(t) >= ch.start_s, 1.0, 0.0)
        return np.clip((np.asarray(t) - ch.start_s) / ch.ramp_s, 0.0, 1.0)

    def _sum_ramps(self, kind: str, t) -> np.ndarray:
        total = np.zeros_like(np.asarray(t, dtype=float))
        for ch in self.changes:
            if ch.kind == kind:
                total = total + ch.magnitude * self._ramp(ch, t)
        return total

    def _act_scale(self, t_sim: float) -> float:
        return max(0.05, 1.0 + float(self._sum_ramps("busier", t_sim)))

    def _shift_hours(self, day: int) -> float:
        day_start_sim = day * DAY_S - self.start_s
        return sum(ch.magnitude for ch in self.changes
                   if ch.kind == "shifted" and ch.start_s <= day_start_sim)

    # ---- one day's schedule --------------------------------------------------
    def events(self, day: int) -> DayEvents:
        """Generate (deterministically) the event schedule for calendar day `day`."""
        c = self.cfg
        rng = np.random.default_rng([self.seed, 1, day])
        ev = DayEvents(day=day)
        base = day * DAY_S
        weekend = (c.start_day_of_week + day) % 7 >= 5
        shift = self._shift_hours(day)

        # presence span
        if weekend:
            if rng.random() >= c.weekend_presence_prob:
                return ev
            arr = rng.normal(11.0, 1.5)
            dep = arr + max(1.0, rng.normal(3.0, 1.0))
            n_breaks = int(rng.poisson(0.7))
        else:
            if rng.random() < c.absent_prob:
                return ev
            arr = rng.normal(c.arrive_mean_h, c.arrive_sd_h)
            dep = max(rng.normal(c.leave_mean_h, c.leave_sd_h), arr + 3.0)
            n_breaks = int(rng.poisson(c.breaks_per_day))
        arr = float(np.clip(arr + shift, 0.5, 21.5))
        dep = float(np.clip(dep + shift, arr + 1.0, 23.5))
        A, D = base + arr * 3600.0, base + dep * 3600.0

        # absences inside the span (lunch + short breaks)
        away: list[tuple[float, float]] = []
        if not weekend and rng.random() < c.lunch_prob:
            ls = base + (c.lunch_mean_h + shift + rng.normal(0.0, c.lunch_sd_h)) * 3600.0
            ld = float(np.clip(rng.normal(c.lunch_dur_mean_min, c.lunch_dur_sd_min), 15, 90)) * 60.0
            away.append((ls, ls + ld))
        if D - A > 7200.0:
            for _ in range(n_breaks):
                bs = rng.uniform(A + 1800.0, D - 1800.0)
                bd = float(np.clip(rng.exponential(c.break_dur_mean_min), 1, 25)) * 60.0
                away.append((bs, bs + bd))
        clipped = [(max(a, A + 900.0), min(b, D - 900.0)) for a, b in away]
        away = _merge([(a, b) for a, b in clipped if b - a >= 60.0], gap=120.0)

        cur = A
        for a, b in away:
            if a - cur >= 120.0:
                ev.sessions.append((cur, a))
            cur = b
        if D - cur >= 120.0:
            ev.sessions.append((cur, D))

        hour_factor = rng.lognormal(0.0, c.activity_hour_sigma, size=24)

        # per-session events
        for s, e in ev.sessions:
            ev.door.append((s, s + rng.uniform(*c.entry_open_s)))                    # entering
            ev.motion.append((s + 0.5, s + 0.5 + rng.uniform(8.0, 25.0)))
            ev.presence.append((s + 1.0, e))
            ev.door.append((e - 0.5, e - 0.5 + rng.uniform(*c.exit_open_s)))         # leaving
            ev.motion.append((e - rng.uniform(6.0, 15.0), e))
            thr = max(0.05, float(rng.normal(c.lamp_threshold_mean, c.lamp_threshold_sd)))
            ev.lamp.append((s + 3.0, e - 2.0, thr))

            # desk activity bursts, Poisson with an hour-varying rate
            tc, lim = s + 30.0, e - 20.0
            while True:
                h = min(23, int((tc - base) // 3600.0))
                rate = c.burst_rate_per_h * self._act_scale(tc - self.start_s) * hour_factor[h]
                tc += rng.exponential(3600.0 / max(rate, 1e-6))
                if tc >= lim:
                    break
                dur = float(np.clip(rng.exponential(c.burst_mean_s), 3.0, 120.0))
                ev.motion.append((tc, min(tc + dur, lim)))
                tc += dur

        # visitors and propped-open doors (weekdays only)
        if not weekend and ev.sessions:
            long_sessions = [(s, e) for s, e in ev.sessions if e - s > 900.0]
            if long_sessions:
                weights = np.array([e - s for s, e in long_sessions])
                weights = weights / weights.sum()
                noon_scale = self._act_scale(base + 12 * 3600.0 - self.start_s)
                for _ in range(int(rng.poisson(c.visitors_per_day * noon_scale))):
                    s, e = long_sessions[int(rng.choice(len(long_sessions), p=weights))]
                    v = rng.uniform(s + 300.0, e - 600.0)
                    stay = min(rng.uniform(120.0, 480.0), e - 60.0 - v)
                    if stay < 60.0:
                        continue
                    ev.visitors += 1
                    ev.door.append((v, v + rng.uniform(4.0, 10.0)))
                    ev.motion.append((v + 0.5, v + 0.5 + rng.uniform(10.0, 25.0)))
                    tc = v + 10.0
                    while True:
                        tc += rng.exponential(3600.0 / c.visitor_burst_rate_per_h)
                        if tc >= v + stay - 10.0:
                            break
                        dur = float(np.clip(rng.exponential(20.0), 3.0, 90.0))
                        ev.motion.append((tc, min(tc + dur, v + stay - 10.0)))
                        tc += dur
                    ev.door.append((v + stay, v + stay + rng.uniform(3.0, 7.0)))
                    ev.motion.append((v + stay - 8.0, v + stay + 2.0))

                if rng.random() < c.propped_door_prob:
                    s, e = long_sessions[int(rng.choice(len(long_sessions), p=weights))]
                    if e - s > 1800.0:
                        t0 = rng.uniform(s + 600.0, e - 1200.0)
                        dur = min(rng.uniform(300.0, 1800.0), e - t0 - 60.0)
                        ev.propped += 1
                        ev.motion.append((t0, t0 + 8.0))
                        ev.door.append((t0, t0 + dur))
        return ev

    # ---- the simulation -------------------------------------------------------
    def states(self, duration_s: float) -> Iterator[WorldState]:
        """Yield one WorldState per tick for `duration_s` seconds. Deterministic for a given seed."""
        c = self.cfg
        dt = c.sample_period_s
        n = int(round(duration_s / dt))
        if n <= 0:
            return
        start_s = self.start_s
        n_days = int(math.ceil((start_s + n * dt) / DAY_S))

        # 1. event schedule -> boolean masks over the whole run
        occupied = np.zeros(n, dtype=bool)
        motion = np.zeros(n, dtype=bool)
        door = np.zeros(n, dtype=bool)
        lamp_windows: list[tuple[int, int, float]] = []
        for day in range(n_days):
            ev = self.events(day)
            _paint(occupied, ev.presence, start_s, dt)
            _paint(motion, ev.motion, start_s, dt)
            _paint(door, ev.door, start_s, dt)
            for a, b, thr in ev.lamp:
                lamp_windows.append((max(0, _idx(a, start_s, dt)), min(n, _idx(b, start_s, dt)), thr))

        # 1b. abrupt natural light changes, per-day seeded so a longer run extends a shorter one
        abrupt = None
        if c.cloud_events_per_h > 0.0:
            abrupt = np.ones(n)
            for day in range(n_days):
                rng_c = np.random.default_rng([self.seed, 4, day])
                day_h = c.sunset_h - c.sunrise_h
                for _ in range(int(rng_c.poisson(c.cloud_events_per_h * day_h))):
                    a = day * DAY_S + (c.sunrise_h + rng_c.uniform(0.0, day_h)) * 3600.0
                    b = a + rng_c.uniform(*c.cloud_duration_range_s)
                    f = rng_c.uniform(*c.cloud_drop_range)
                    i0, i1 = max(0, _idx(a, start_s, dt)), min(n, _idx(b, start_s, dt))
                    if i1 > i0:
                        abrupt[i0:i1] = np.minimum(abrupt[i0:i1], f)

        # 2. slow weather and cloud processes on a 5-minute grid (clock-based, prefix-stable)
        step = 300.0
        m = int(n_days * DAY_S / step) + 2
        grid = np.arange(m) * step
        w_temp = _ou(np.random.default_rng([self.seed, 2, 0]), m, step, c.weather_tau_h * 3600.0, c.weather_sd_c)
        w_rh = _ou(np.random.default_rng([self.seed, 2, 1]), m, step, c.weather_tau_h * 3600.0, c.rh_weather_sd)
        w_cloud = _ou(np.random.default_rng([self.seed, 2, 2]), m, step, c.cloud_tau_h * 3600.0, 1.0)
        cloud_series = np.clip(0.8 + 0.25 * w_cloud, c.cloud_min, 1.0)

        inv_tau = 1.0 / c.tau_room_s
        q_occ = c.occ_gain_c / c.tau_room_s
        q_lamp = c.lamp_gain_c / c.tau_room_s
        inv_td = 1.0 / c.tau_door_s
        inv_th = 1.0 / c.tau_rh_s
        sqdt = math.sqrt(dt)

        chunk = max(1, int(round(c.chunk_s / dt)))
        T = H = 0.0
        for ci, i0 in enumerate(range(0, n, chunk)):
            i1 = min(n, i0 + chunk)
            nc = i1 - i0
            t = np.arange(i0, i1) * dt
            clock = start_s + t
            hod = (clock / 3600.0) % 24.0

            # light
            sun = np.clip(np.sin(np.pi * (hod - c.sunrise_h) / (c.sunset_h - c.sunrise_h)), 0.0, None) ** c.sun_exponent
            gain = c.daylight_peak * np.clip(1.0 + self._sum_ramps("brighter", t), 0.0, None)
            env0 = gain * sun
            cloud = np.interp(clock, grid, cloud_series)

            lamp = np.zeros(nc, dtype=bool)
            for a, b, thr in lamp_windows:
                lo, hi = max(a, i0), min(b, i1)
                if hi > lo:
                    lamp[lo - i0:hi - i0] = env0[lo - i0:hi - i0] < thr

            occ = occupied[i0:i1]
            dr = door[i0:i1]
            rng = np.random.default_rng([self.seed, 3, ci])
            mo = motion[i0:i1] | (occ & (rng.random(nc) < c.fidget_p))
            day_light = env0 * cloud if abrupt is None else env0 * cloud * abrupt[i0:i1]
            light = day_light + c.lamp_level * lamp + c.corridor_level * dr

            # outdoor temperature drives the room
            warm = self._sum_ramps("warmer", t)
            tout = (c.out_mean_c
                    + c.out_amp_c * np.sin(2.0 * np.pi * (hod - (c.out_peak_h - 6.0)) / 24.0)
                    + np.interp(clock, grid, w_temp) + warm)
            rh_w = np.interp(clock, grid, w_rh)

            if ci == 0:
                T = float(c.out_mean_c + np.interp(clock[0], grid, w_temp) + warm[0])
                H = float(c.rh_base + rh_w[0] - c.rh_temp_coeff * (T - c.t_ref_c))

            noise_t = (rng.standard_normal(nc) * c.temp_noise * sqdt).tolist()
            noise_h = (rng.standard_normal(nc) * c.rh_noise * sqdt).tolist()
            tout_l, rhw_l = tout.tolist(), rh_w.tolist()
            occ_l, door_l, lamp_l = occ.astype(np.int8).tolist(), dr.astype(np.int8).tolist(), lamp.astype(np.int8).tolist()

            temps = [0.0] * nc
            hums = [0.0] * nc
            for j in range(nc):
                d = door_l[j]
                T += dt * ((tout_l[j] - T) * inv_tau + q_occ * occ_l[j] + q_lamp * lamp_l[j]
                           + (c.corridor_temp_c - T) * inv_td * d) + noise_t[j]
                target = c.rh_base + rhw_l[j] + c.rh_occ_gain * occ_l[j] - c.rh_temp_coeff * (T - c.t_ref_c)
                H += dt * ((target - H) * inv_th + (c.rh_corridor - H) * inv_td * d) + noise_h[j]
                H = min(99.0, max(5.0, H))
                temps[j] = T
                hums[j] = H

            # ground-truth regime tags (first listed change wins on overlap)
            kinds: list[Optional[str]] = [None] * nc
            for ch in reversed(self.changes):
                a = max(i0, _idx(ch.start_s, 0.0, dt))
                b = min(i1, _idx(ch.label_end_s, 0.0, dt))
                if b > a:
                    kinds[a - i0:b - i0] = [ch.kind] * (b - a)

            for row in zip(t.tolist(), hod.tolist(), temps, hums, occ.tolist(), mo.tolist(),
                           dr.tolist(), lamp.tolist(), light.tolist(), kinds):
                yield WorldState(*row)