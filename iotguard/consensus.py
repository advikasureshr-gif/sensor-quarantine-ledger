"""
consensus.py - cross-sensor consensus and the trust gate: the heart of the project.

    WindowFeatures --ConsensusEngine--> ConsensusResult --Gate(+ Mahalanobis score D)--> Decision

ConsensusEngine looks only at fixed physical relationships between sensors; it never uses the
learned mu/Sigma. It is stateful (it tracks who is believed to be in the room) and must see
consecutive windows in order.

Occupancy tracking
------------------
    UNKNOWN   at start. First PIR activity -> OCCUPIED; silence for `vacancy_timeout_s` -> EMPTY.
    EMPTY     a door opening followed by PIR motion (an "entry") -> OCCUPIED.
    OCCUPIED  a further such passage is exit-like (an exit or a visitor); if the PIR then stays
              silent for `vacancy_timeout_s` the room is believed EMPTY. Motion shortly after the
              passage means someone stayed, so the exit suspicion is dropped.
Motion while OCCUPIED is normal. Motion while EMPTY with no door opening to explain it is the
signature of a spoofed or stuck PIR, a masked door, or a replay.

Evidence (each is positive or negative, see ConsensusConfig for the thresholds)
    door_pir_ok / door_pir_missing           door opening vs motion
    door_light_ok / door_light_missing        door opening vs corridor-light step
    pir_unexplained                           motion in an empty room, no door opening
    pir_stuck                                 PIR continuously high far too long
    temp_hum_inconsistent                     abrupt temperature step without a humidity response
    light_unexplained                         abrupt light step in an empty room, no door opening

Gate
----
Combines the evidence with the Mahalanobis score D:
    SUSPICIOUS  negative evidence >= suspicious_neg                         -> freeze (alpha = 0)
    UNCERTAIN   weaker contradiction, or an unexplained statistical outlier  -> slow learning
    TRUSTED     no contradiction and D within threshold, OR a corroborated event
                (positive evidence) with a not-absurd D                      -> full learning
A corroborated event may legitimately score high (a real door entry has D around 8), so
corroboration overrides a high D; a contradiction overrides everything.
Suspicion memory: contradictions raise a decaying level; while it is high, windows that are
still anomalous stay SUSPICIOUS instead of being slowly learned, until the anomaly ends or
`suspicion_max_hold_windows` pass.

Known limit, by design: an attack that keeps every checked relationship plausible produces no
contradiction and is treated like a legitimate change (UNCERTAIN, slow learning).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Optional

from .config import DEFAULT_CONFIG, ConsensusConfig, FeatureConfig, OnlineStatsConfig
from .features import WindowFeatures


# Which family each piece of evidence belongs to, and which contradictions are STRONG. With the
# families evidence model (ConsensusConfig.evidence), a window is SUSPICIOUS on one strong
# contradiction or on weak ones from two different families; light is weak by nature.
FAMILY = {
    "door_pir_ok": "door_pir", "door_pir_missing": "door_pir",
    "door_light_ok": "light", "door_light_missing": "light", "light_unexplained": "light",
    "pir_unexplained": "occupancy", "pir_unexplained_rate": "occupancy", "pir_unverified": "occupancy",
    "pir_stuck": "pir_shape", "temp_hum_inconsistent": "dht",
}
STRONG = frozenset({"door_pir_missing", "pir_unexplained", "pir_unexplained_rate", "pir_stuck", "temp_hum_inconsistent"})


class Trust(str, Enum):
    TRUSTED = "TRUSTED"
    UNCERTAIN = "UNCERTAIN"
    SUSPICIOUS = "SUSPICIOUS"


class Occupancy(str, Enum):
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    UNSURE = "UNSURE"            # confidence-based tracker only: between EMPTY and OCCUPIED
    OCCUPIED = "OCCUPIED"


@dataclass(frozen=True)
class ConsensusResult:
    index: int
    occupancy: Occupancy                        # believed state at the end of the window
    evidence: tuple[tuple[str, float, str], ...]   # (name, signed value, human-readable detail)
    pos: float                                  # sum of positive evidence
    neg: float                                  # sum of negative evidence (<= 0)
    occ_conf: float = math.nan                  # occupancy confidence at the end of the window (nan = legacy tracker)

    @property
    def contradictions(self) -> tuple[str, ...]:
        return tuple(n for n, v, _ in self.evidence if v < 0)

    @property
    def corroborations(self) -> tuple[str, ...]:
        return tuple(n for n, v, _ in self.evidence if v > 0)


@dataclass(frozen=True)
class Decision:
    index: int
    label: Trust
    reason: str
    d: float
    suspicion: float                            # gate's suspicion level AFTER this window
    occupancy: Occupancy

    def alpha(self, cfg: OnlineStatsConfig = DEFAULT_CONFIG.online_stats) -> float:
        """The learning rate this decision allows."""
        return {Trust.TRUSTED: cfg.alpha_trusted, Trust.UNCERTAIN: cfg.alpha_uncertain,
                Trust.SUSPICIOUS: cfg.alpha_suspicious}[self.label]


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
class ConsensusEngine:
    HIST_S = 90.0          # seconds of PIR/LDR history kept for look-back checks

    def __init__(self, cfg: ConsensusConfig = DEFAULT_CONFIG.consensus,
                 features_cfg: FeatureConfig = DEFAULT_CONFIG.features,
                 sample_period_s: float = DEFAULT_CONFIG.sampling.sample_period_s) -> None:
        if cfg.light_check not in ("off", "positive_only", "full"):
            raise ValueError("light_check must be 'off', 'positive_only' or 'full'")
        if features_cfg.hop_s != features_cfg.window_s:
            raise ValueError("ConsensusEngine requires disjoint windows (hop_s == window_s)")
        self.cfg = cfg
        self._dt = sample_period_s
        self._occ = cfg.occupancy
        self._ev = cfg.evidence
        self.reset()

    def reset(self) -> None:
        self.state = Occupancy.UNKNOWN
        self._first_t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._last_motion_t: Optional[float] = None
        self._exit_candidate = False
        self._passage_t = -math.inf
        self._hist: deque[tuple[float, int, int]] = deque()      # (t, pir, ldr_raw)
        self._opens: deque[float] = deque()                       # recent confirmed door openings
        self._pending_doors: list[float] = []
        self._pending_steps: list[float] = []
        self._pir_prev = 0
        self._pir_start = 0.0
        self._pir_explained = False
        self._prev_ldr: Optional[int] = None
        self._prev_th: Optional[tuple[float, float, int]] = None  # (temp, hum, window index)
        self._conf = 0.0                                          # occupancy confidence (confidence-based tracker)
        self._t_ind = 0.0                                         # time of the last independent evidence
        self._exit_pending = False
        # evidence model state (only used when cfg.evidence is set)
        self._calib = True                    # learning this room's relationships until end_calibration()
        self._n_light_step = self._n_light_nostep = 0
        self._empty_s = self._unexp_s = 0.0   # background: unexplained PIR seconds per second of believed-empty time
        self._bucket = 0.0                    # leaky bucket of unexplained PIR seconds
        self._light_enforced = False
        self._bucket_thr = math.inf

    def end_calibration(self) -> None:
        """
        Call once, after the calibration windows have been evaluated. Fixes what this room taught the
        engine (evidence model only): whether the door->light relationship is reliable, and how many
        unexplained PIR seconds its own false triggers produce. Until then the light check is not enforced
        and the temporal bucket is inactive.
        """
        if self._ev is None or not self._calib:
            return
        e = self._ev
        self._calib = False
        n = self._n_light_step + self._n_light_nostep
        self._light_enforced = n >= e.light_min_events and self._n_light_step / n >= e.light_reliable_min
        background = self._unexp_s / self._empty_s if self._empty_s > 0 else 0.0
        self._bucket_thr = max(e.bucket_threshold_s, e.bucket_background_k * background * e.bucket_tau_s)

    @property
    def light_enforced(self) -> bool:
        return self._light_enforced

    @property
    def bucket_threshold_s(self) -> float:
        return self._bucket_thr

    # ---- helpers -------------------------------------------------------------------------
    def _door_explains(self, start: float, now: float) -> bool:
        c = self.cfg
        return any(start - 2.0 <= te <= min(now, start + c.door_pir_post_s) for te in self._opens)

    def _passage(self, te: float) -> None:
        if self.state in (Occupancy.UNKNOWN, Occupancy.EMPTY):      # an entry
            self.state = Occupancy.OCCUPIED
            self._exit_candidate = False
        else:                                                        # exit-like (leaving or a visitor)
            self._exit_candidate = True
            self._passage_t = te

    # ---- confidence-based occupancy (used only when cfg.occupancy is set) -----------------------
    def _relabel(self) -> None:
        o = self._occ
        self.state = (Occupancy.OCCUPIED if self._conf >= o.occupied_threshold
                      else Occupancy.EMPTY if self._conf <= o.empty_threshold else Occupancy.UNSURE)

    def _decay(self, t: float, dt: float) -> None:
        o = self._occ
        if self.state is Occupancy.UNKNOWN:
            return
        if self._exit_pending:                          # silence after an exit-like passage
            self._conf *= math.exp(-dt / o.exit_decay_tau_s)
        elif t - self._t_ind > o.pir_horizon_s:         # horizon expired: motion can no longer hold the belief up
            self._conf *= math.exp(-dt / o.expired_decay_tau_s)
        else:
            return
        self._relabel()
        if self._exit_pending and self.state is Occupancy.EMPTY:
            self._exit_pending = False                  # fully decayed: motion can no longer revive the belief

    def _silent_before(self, te: float) -> bool:
        """No PIR activity in the entry_silence_s before a passage (excluding its own first seconds)."""
        lo = te - self._occ.entry_silence_s
        return not any(p for tt, p, _ in self._hist if lo <= tt <= te - 3.0)

    def _motion_conf(self, t: float) -> None:
        o = self._occ
        if self.state is Occupancy.UNKNOWN:             # no history yet: assume someone is here, start the clock
            self._conf, self._t_ind = o.pir_cap, t
            self._relabel()
        elif self._exit_pending:
            if t - self._passage_t >= self.cfg.exit_clear_s:      # motion well after the passage: someone stayed
                self._exit_pending = False
                self._conf = max(self._conf, o.pir_cap)
                self._t_ind = self._passage_t
                self._relabel()
        elif self.state is not Occupancy.EMPTY and t - self._t_ind <= o.pir_horizon_s:
            self._conf = max(self._conf, o.pir_cap)     # PIR alone: capped, and only within the horizon
            self._relabel()

    def _passage_conf(self, te: float) -> None:
        o = self._occ
        if self._conf >= o.occupied_threshold or self._exit_pending or not self._silent_before(te):
            # someone was believed inside, or was moving just before the door (a stale belief): exit-like
            self._conf = min(self._conf, o.exit_conf)
            self._exit_pending, self._passage_t = True, te
        else:                                                             # silence before: an entry, independent evidence
            self._conf, self._t_ind, self._exit_pending = 1.0, te, False
        self._relabel()

    def _light_step(self, te: float) -> Optional[bool]:
        before = [l for tt, _, l in self._hist if te - 4.0 <= tt <= te - 1.5]
        after = [l for tt, _, l in self._hist if te <= tt <= te + 3.0]
        if len(before) < 2 or len(after) < 2:
            return None
        return median(after) - median(before) >= self.cfg.door_light_step_adc

    def _temp_hum(self, w: WindowFeatures) -> Optional[str]:
        c = self.cfg
        if "temp_mean" not in w.names or "hum_mean" not in w.names:
            return None
        t, h = w.feature("temp_mean"), w.feature("hum_mean")
        if not (w.valid and not w.dht_carried and math.isfinite(t) and math.isfinite(h)):
            return None
        out = None
        prev = self._prev_th
        if prev is not None and w.index == prev[2] + 1:
            dT, dH = t - prev[0], h - prev[1]
            if abs(dT) >= c.temp_step_c or abs(dH) >= c.hum_step_pct:
                lo, hi = c.hum_per_c_range
                ratio = (-dH / dT) if dT != 0 else math.inf
                if not lo <= ratio <= hi:
                    out = f"temperature moved {dT:+.1f} C while humidity moved {dH:+.1f} %RH"
        self._prev_th = (t, h, w.index)
        return out

    # ---- one window ---------------------------------------------------------------------------
    def evaluate(self, w: WindowFeatures) -> ConsensusResult:
        if not w.samples:
            raise ValueError("window has no samples; extract windows with keep_samples=True")
        c = self.cfg
        n_ok = n_bad = n_step = n_nostep = n_unexplained_steps = 0
        unexplained_s = unverified_s = 0.0

        for s in w.samples:
            t = s.t
            dt = min(max(t - self._prev_t, 0.0), 2 * self._dt) if self._prev_t is not None else self._dt
            unexp_dt = 0.0
            if self._first_t is None:
                self._first_t = t
            if self._occ is not None:
                self._decay(t, dt)
            if s.door_event == 1:
                self._opens.append(t)
                self._pending_doors.append(t)

            # --- PIR bookkeeping
            if s.pir:
                if not self._pir_prev:                                # a new high interval starts
                    self._pir_start = t - s.pir_high_s
                    self._pir_explained = False
                if self._occ is None:
                    if self.state is Occupancy.UNKNOWN:
                        self.state = Occupancy.OCCUPIED
                    if not self._pir_explained:
                        self._pir_explained = self.state is Occupancy.OCCUPIED or self._door_explains(self._pir_start, t)
                    if not self._pir_explained:
                        unexplained_s += dt
                        unexp_dt = dt
                    self._last_motion_t = t
                    if self._exit_candidate and t - self._passage_t >= c.exit_clear_s:
                        self._exit_candidate = False                   # someone stayed inside
                else:
                    self._motion_conf(t)
                    if not self._pir_explained:
                        self._pir_explained = self.state is Occupancy.OCCUPIED or self._door_explains(self._pir_start, t)
                    if not self._pir_explained:
                        if self.state is Occupancy.EMPTY:
                            unexplained_s += dt                        # strong: nobody can be here
                            unexp_dt = dt
                        elif self.state is Occupancy.UNSURE:
                            unverified_s += dt                         # weak: belief has expired but not emptied
                    self._last_motion_t = t
            self._pir_prev = s.pir

            # --- evidence model: background and leaky bucket of unexplained PIR seconds
            if self._ev is not None:
                if self._calib:
                    self._unexp_s += unexp_dt
                    if self.state is Occupancy.EMPTY:
                        self._empty_s += dt
                self._bucket = self._bucket * math.exp(-dt / self._ev.bucket_tau_s) + unexp_dt

            # --- abrupt light step
            if self._prev_ldr is not None and abs(s.ldr_raw - self._prev_ldr) >= c.abrupt_light_step_adc:
                self._pending_steps.append(t)
            self._prev_ldr = s.ldr_raw

            self._hist.append((t, s.pir, s.ldr_raw))
            while self._hist and self._hist[0][0] < t - self.HIST_S:
                self._hist.popleft()
            while self._opens and self._opens[0] < t - 60.0:
                self._opens.popleft()

            # --- judge door openings once their look-ahead is complete
            while self._pending_doors and self._pending_doors[0] + c.door_pir_post_s <= t:
                te = self._pending_doors.pop(0)
                if any(p for tt, p, _ in self._hist if te - c.door_pir_pre_s <= tt <= te + c.door_pir_post_s):
                    n_ok += 1
                    if self._occ is None:
                        self._passage(te)
                    else:
                        self._passage_conf(te)
                else:
                    n_bad += 1
                step = self._light_step(te)
                if self._ev is not None and self._calib and step is not None:
                    if step:
                        self._n_light_step += 1
                    else:
                        self._n_light_nostep += 1
                if step is True:
                    n_step += 1
                elif step is False:
                    n_nostep += 1

            # --- judge abrupt light steps
            while self._pending_steps and self._pending_steps[0] + c.abrupt_light_post_s <= t:
                ts = self._pending_steps.pop(0)
                if self.state is Occupancy.EMPTY and not any(ts - 6.0 <= te <= ts + c.abrupt_light_post_s for te in self._opens):
                    n_unexplained_steps += 1

            # --- vacancy
            if self._occ is None and self.state is Occupancy.OCCUPIED and self._exit_candidate:
                ref = max(self._last_motion_t if self._last_motion_t is not None else -math.inf, self._passage_t)
                if t - ref >= c.vacancy_timeout_s:
                    self.state, self._exit_candidate = Occupancy.EMPTY, False
            elif self.state is Occupancy.UNKNOWN and t - self._first_t >= c.vacancy_timeout_s:
                self.state = Occupancy.EMPTY
                self._conf = 0.0
            self._prev_t = t

        # --- assemble evidence
        ev: list[tuple[str, float, str]] = []
        if n_ok:
            ev.append(("door_pir_ok", c.w_door_pir * min(n_ok, 2), f"{n_ok} door opening(s) followed by motion"))
        if n_bad:
            ev.append(("door_pir_missing", -c.w_door_pir * min(n_bad, 2), f"{n_bad} door opening(s) with no motion"))
        light_mode = c.light_check
        if self._ev is not None and self._ev.auto_light:
            light_mode = "full" if self._light_enforced else "positive_only"      # learned from this room
        if light_mode != "off":
            if n_step:
                ev.append(("door_light_ok", c.w_light * min(n_step, 2), f"{n_step} opening(s) with a light step"))
            if n_nostep and light_mode == "full":
                ev.append(("door_light_missing", -c.w_light * min(n_nostep, 2), f"{n_nostep} opening(s) with no light step"))
        if unexplained_s >= c.pir_unexplained_min_s:
            ev.append(("pir_unexplained", -c.w_pir_unexplained,
                       f"{unexplained_s:.1f} s of motion with nobody believed inside and no door opening"))
        if (self._ev is not None and self._ev.temporal_bucket and self._bucket >= self._bucket_thr
                and not any(n == "pir_unexplained" for n, _, _ in ev)):
            ev.append(("pir_unexplained_rate", -c.w_pir_unexplained,
                       f"unexplained motion keeps accumulating in an empty room ({self._bucket:.0f} s, limit {self._bucket_thr:.0f} s)"))
        if self._occ is not None and unverified_s >= c.pir_unexplained_min_s:
            ev.append(("pir_unverified", -self._occ.w_pir_unverified,
                       f"{unverified_s:.1f} s of motion while occupancy confidence is low and no door opening explains it"))
        if w.pir_high_max_s >= c.pir_max_continuous_s:
            ev.append(("pir_stuck", -c.w_pir_stuck, f"PIR continuously high for {w.pir_high_max_s:.0f} s"))
        if n_unexplained_steps:
            ev.append(("light_unexplained", -c.w_abrupt_light, f"{n_unexplained_steps} abrupt light step(s) in an empty room"))
        th = self._temp_hum(w)
        if th:
            ev.append(("temp_hum_inconsistent", -c.w_temp_hum, th))

        pos = sum(v for _, v, _ in ev if v > 0)
        neg = sum(v for _, v, _ in ev if v < 0)
        return ConsensusResult(w.index, self.state, tuple(ev), pos, neg, self._conf if self._occ is not None else math.nan)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
class Gate:
    def __init__(self, cfg: ConsensusConfig = DEFAULT_CONFIG.consensus,
                 d_threshold: float = DEFAULT_CONFIG.anomaly.mahalanobis_threshold) -> None:
        self.cfg = cfg
        self.d_threshold = d_threshold
        self.suspicion = 0.0
        self._held_run = 0

    def reset(self) -> None:
        self.suspicion, self._held_run = 0.0, 0

    def _negative_verdict(self, cons: ConsensusResult) -> tuple[Optional[Trust], str]:
        """What the contradictions alone say. (None, "") when there are none."""
        c = self.cfg
        if c.evidence is not None and c.evidence.families:
            strong = [n for n, v, _ in cons.evidence if v < 0 and n in STRONG]
            weak = {FAMILY[n]: n for n, v, _ in cons.evidence if v < 0 and n not in STRONG}
            if strong:
                return Trust.SUSPICIOUS, "contradiction: " + ", ".join(strong)
            if len(weak) >= 2:
                return Trust.SUSPICIOUS, "independent weak contradictions: " + ", ".join(weak.values())
            if weak:
                return Trust.UNCERTAIN, "weak contradiction: " + ", ".join(weak.values())
            return None, ""
        if cons.neg <= -c.suspicious_neg:
            return Trust.SUSPICIOUS, "contradiction: " + ", ".join(cons.contradictions)
        if cons.neg < 0:
            return Trust.UNCERTAIN, "weak contradiction: " + ", ".join(cons.contradictions)
        return None, ""

    def decide(self, cons: ConsensusResult, d: float) -> Decision:
        c = self.cfg
        anomalous = d > self.d_threshold
        self.suspicion *= c.suspicion_decay
        if self.suspicion < c.suspicion_hold:
            self._held_run = 0

        # 1. classify from the evidence
        verdict, why = self._negative_verdict(cons)
        if verdict is Trust.SUSPICIOUS:
            label, reason = Trust.SUSPICIOUS, why
        elif verdict is Trust.UNCERTAIN:
            label, reason = Trust.UNCERTAIN, why
        elif not anomalous:
            label, reason = Trust.TRUSTED, "consistent" if not cons.corroborations else "consistent, corroborated"
        elif cons.pos >= c.corroborated_min and d <= c.corroborated_max_d:
            label, reason = Trust.TRUSTED, "corroborated event: " + ", ".join(cons.corroborations)
        else:
            label, reason = Trust.UNCERTAIN, "unexplained statistical outlier"

        # 2. suspicion memory
        if label is Trust.SUSPICIOUS:
            self.suspicion = min(c.suspicion_cap, self.suspicion + 1.0)
            self._held_run = 0
        elif label is Trust.UNCERTAIN and anomalous and self.suspicion >= c.suspicion_hold:
            if self._held_run < c.suspicion_max_hold_windows:
                label, reason = Trust.SUSPICIOUS, "held: anomaly persists after an earlier contradiction"
                self.suspicion = min(c.suspicion_cap, self.suspicion + c.suspicion_refresh)
                self._held_run += 1
            else:
                reason += " (suspicion hold expired)"
                self.suspicion = 0.0
        return Decision(cons.index, label, reason, d, self.suspicion, cons.occupancy)