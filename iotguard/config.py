"""
config.py - every tunable constant for the simulator and the pipeline, in one place.

Nothing else in the project should hard-code a threshold, window length, or
sensor pin-equivalent constant. When we retune against real ESP32 data later,
this is the only file that should need editing.

Grouped to match the pipeline stages in the flow diagram:
    sensor readings -> preprocessing -> feature extraction -> feature vector
    -> (calibration | normal operation: Mahalanobis -> consensus -> gate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    # Matches SAMPLE_PERIOD_MS in esp32_sensor_hub.ino. If you change the
    # firmware's rate, change it here too, or windows will be mis-sized.
    sample_period_s: float = 0.5          # 2 Hz, i.e. one Reading every 0.5 s


# ---------------------------------------------------------------------------
# Preprocessing (raw Reading -> cleaned Reading, before windowing)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    # DHT: hold the last good value across this many seconds of nulls before
    # treating temp/hum as missing for feature purposes. The DHT22 itself
    # only refreshes every ~2 s, so a few consecutive nulls is normal, not a fault.
    dht_hold_max_s: float = 10.0

    # DHT values outside these ranges are physically impossible (sensor glitch) and are
    # treated as a failed read. Deliberately WIDE: preprocessing must never "clean away"
    # an attack such as a sudden +8 C spike; deciding what is anomalous is the detector's job.
    temp_valid_range_c: tuple[float, float] = (-10.0, 60.0)
    hum_valid_range: tuple[float, float] = (0.0, 100.0)

    # PIR: ignore new "motion" triggers for this many seconds after one fires
    # (HC-SR501 retriggers/chatters near the edge of detection). Mirrors, but
    # is independent of, any debounce already done in firmware.
    pir_retrigger_lockout_s: float = 2.0

    # Door (reed switch): a new state is accepted only once it has been seen CONTINUOUSLY
    # for this long, measured from its first sample to the current one. At 2 Hz:
    # 0.5 = 2 consecutive samples, 1.0 = 3 consecutive samples. Filters contact chatter
    # while the door closes, at the cost of reporting a transition this much later.
    door_debounce_s: float = 1.0

    # LDR: exponential smoothing factor applied to the raw ADC value before
    # it reaches feature extraction (1.0 = no smoothing, lower = smoother).
    ldr_ema_alpha: float = 0.3

    # If two readings are further apart than this (dropped lines, serial reconnect), all
    # filter state is reset instead of being carried across the gap, so a door that
    # changed during the gap does not produce a bogus event.
    gap_reset_s: float = 30.0


# ---------------------------------------------------------------------------
# Feature extraction (cleaned Readings -> one feature vector per window)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    # Window length and hop. hop < window_s gives overlapping windows; set
    # hop_s == window_s for disjoint windows. 30 s / 30 s to start, per the
    # project summary; shorten both for live demos (see DemoFeatureConfig below).
    window_s: float = 30.0
    hop_s: float = 30.0

    # Feature order is fixed and used positionally everywhere downstream:
    #   [temp_mean, hum_mean, pir_count, pir_duty, door_count, ldr_mean, ldr_std]
    # pir_count = motion events (rising edges); pir_duty = fraction of the window the PIR was
    # high. Both are needed: a PIR stuck HIGH has no rising edges, so pir_count alone would
    # read it as silence. door_count = confirmed door transitions (opens + closes).
    feature_names: tuple[str, ...] = (
        "temp_mean", "hum_mean", "pir_count", "pir_duty", "door_count", "ldr_mean", "ldr_std",
    )

    # A window with fewer valid DHT samples than this fraction gets
    # temp_mean/hum_mean carried forward from the previous window instead of
    # computed fresh (avoids a feature vector built from near-zero DHT data).
    min_dht_fraction: float = 0.5

    # A window with fewer samples than this fraction of the expected count (window_s divided
    # by the sample period) is flagged invalid and must not be used for learning or scoring.
    min_window_coverage: float = 0.8


# Shorter windows for the tabletop demo, where you can't wait 30s per sample
# during a live poke-the-sensors demonstration. Swap this in for FeatureConfig
# at the pipeline's construction site, not by editing FeatureConfig itself.
@dataclass(frozen=True)
class DemoFeatureConfig(FeatureConfig):
    window_s: float = 5.0
    hop_s: float = 5.0


# ---------------------------------------------------------------------------
# Calibration (admin-authenticated baseline construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    # Need at least this many trusted windows before mu/Sigma may be built.
    min_trusted_windows: int = 50

    # Robust screening: a candidate window is dropped before the consensus
    # check if any feature is more than this many MADs from the running
    # median of candidates so far (median absolute deviation, robust to
    # outliers unlike mean/std).
    mad_outlier_threshold: float = 3.5

    # Calibration gives up (and alerts the admin) after this many candidate
    # windows if min_trusted_windows still hasn't been reached.
    max_candidate_windows: int = 400


# ---------------------------------------------------------------------------
# Online mu / Sigma estimation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OnlineStatsConfig:
    # Learning rate per consensus decision. Rates are per WINDOW, not per second: with 30 s
    # windows, alpha 0.002 is a time constant of ~4 hours and 0.0004 ~21 hours.
    #   d = x - mu;   mu += alpha * d;   Sigma = (1 - alpha) * (Sigma + alpha * d d^T)
    # Why slow: with alpha 0.05 (10-minute time constant) the baseline tracks the daily cycle, so
    # freezing it for a few hours (after an attack) leaves it stale and the suspicion hold then
    # keeps it frozen: measured 2.4% false SUSPICIOUS windows after a 2 h attack, versus 0.1%
    # at alpha <= 0.002. Zero false alarms on normal data at every rate tested; adaptation to a
    # legitimate +4 C change still completed within days. Placeholders to be tuned by the experiments.
    alpha_trusted: float = 0.002
    alpha_uncertain: float = 0.0004
    alpha_suspicious: float = 0.0       # frozen: no update at all

    # Covariance shrinkage toward its own diagonal: Sigma_used = (1-shrinkage)*Sigma
    # + shrinkage*diag(Sigma). Stabilises the inverse when features are strongly correlated.
    shrinkage: float = 0.1

    # Features put on a log1p scale before modelling. Counts and spreads are heavily skewed;
    # on the raw scale one light step dominates everything (measured: log-scaling ldr_mean
    # takes the covariance condition number from 2e7 to 1e3 and the false-positive rate on
    # normal windows to ~2.2% at D > 4.0).
    log1p_features: tuple[str, ...] = ("pir_count", "door_count", "ldr_mean", "ldr_std")

    # Smallest spread each feature may be treated as having, in the (possibly log1p) units of
    # the model. Added to the covariance diagonal as min_std**2. Without it, a feature that
    # is nearly constant in calibration (e.g. door_count) makes any ordinary deviation look
    # enormous. Measured effect: false positives on normal windows 6.0% -> 2.8% at D > 4.0.
    min_std: dict[str, float] = field(default_factory=lambda: {
        "temp_mean": 0.3,    # deg C
        "hum_mean": 1.0,     # % RH
        "pir_count": 0.5,    # log1p units (one event ~ 0.69)
        "pir_duty": 0.05,    # fraction of the window
        "door_count": 0.5,   # log1p units (one transition ~ 0.69)
        "ldr_mean": 0.3,     # log1p units
        "ldr_std": 0.3,      # log1p units
    })

    # Features left OUT of the Gaussian (still extracted, still available to consensus).
    # Experiment switch: dropping door_count made no measurable difference to false positives.
    exclude_features: tuple[str, ...] = ()

    # Drift cap: after every update, mu is clipped to within this many calibration standard
    # deviations of the calibrated baseline, per feature, so a slow poisoning attack cannot
    # relocate the baseline indefinitely without an admin recalibration. None = no cap.
    # CAUTION: the calibration std includes the normal daily cycle (about 2.2 C for
    # temperature in the simulated room), so 5.0 permits ~11 C of drift: far too loose to stop
    # a 5 C poisoning drift. A tight cap (1-2) stops it but also blocks legitimate shifts until
    # recalibration. This is a real trade-off to SWEEP in the experiments, not a value to trust.
    drift_cap_std: float | None = 5.0


# ---------------------------------------------------------------------------
# Mahalanobis anomaly scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnomalyConfig:
    # Score above this = anomalous. D = sqrt(chi-square(k)) for k features if the features
    # were Gaussian. With k = 7 (checked by simulation): D > 3.0 flags 25% of NORMAL windows,
    # D > 3.5 flags 9%, D > 4.0 flags 2.5%, D > 4.5 flags 0.5%. Real features (counts, door
    # events) are NOT Gaussian and have heavier tails, so the real false-positive rate will
    # be higher than these ideal numbers. Treat 4.0 as a starting point and retune on data.
    # Measured with gaussian.py on two seeds of simulated normal days: 2.2% of held-out normal
    # windows exceed D = 4.0.
    mahalanobis_threshold: float = 4.0


# ---------------------------------------------------------------------------
# Occupancy confidence (used by the consensus engine when ConsensusConfig.occupancy is set)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OccupancyConfig:
    """
    Tracks CONFIDENCE that someone is in the room, separately from the label derived from it.
    The point is to stop motion from reinforcing itself: PIR triggers alone must not keep the
    belief "occupied" alive indefinitely (a spoofed PIR would then explain away all later motion).

      independent evidence   a corroborated door passage. Only this moves the room from
                             empty/unsure to occupied, and only this restarts the horizon clock.
      PIR-only evidence      lifts confidence at most to `pir_cap`, and only while the last
                             independent evidence is no older than `pir_horizon_s`.
      decay                  after an exit-like passage with silence, confidence falls with time
                             constant `exit_decay_tau_s`; once the horizon has expired it falls
                             with `expired_decay_tau_s` however much motion arrives.
    Label: OCCUPIED if confidence >= occupied_threshold, EMPTY if <= empty_threshold, else UNSURE.
    Motion while UNSURE with no door explanation is only WEAK evidence (`w_pir_unverified`), so a
    legitimate stay longer than the horizon costs a mild UNCERTAIN, not a freeze.
    The horizon is the key trade-off: short = more false alarms for long stays, long = a spoofed
    PIR is believed for longer. It is meant to be swept in the experiments, not trusted as given.
    """
    pir_horizon_s: float = 4 * 3600.0
    pir_cap: float = 0.8
    occupied_threshold: float = 0.6
    empty_threshold: float = 0.2
    exit_conf: float = 0.5               # confidence right after an exit-like passage
    exit_decay_tau_s: float = 650.0      # 0.5 -> 0.2 in ~10 min of silence, like the old vacancy timeout
    expired_decay_tau_s: float = 1800.0  # 0.8 -> UNSURE in ~9 min, -> EMPTY in ~42 min
    w_pir_unverified: float = 0.5
    # An ENTRY into a room believed not occupied must be preceded by this much PIR silence. A person
    # leaving has been moving just before the door; if the belief had already decayed during a long
    # stay, that exit must not be mistaken for an arrival. Must stay below the engine's 80 s history.
    entry_silence_s: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 < self.entry_silence_s <= 80.0:
            raise ValueError("entry_silence_s must be in (0, 80]")
        if not 0.0 < self.empty_threshold < self.occupied_threshold < self.pir_cap <= 1.0:
            raise ValueError("need 0 < empty_threshold < occupied_threshold < pir_cap <= 1")
        if not 0.0 < self.exit_conf < self.occupied_threshold:
            raise ValueError("exit_conf must be positive and below occupied_threshold")
        if min(self.pir_horizon_s, self.exit_decay_tau_s, self.expired_decay_tau_s) <= 0:
            raise ValueError("horizon and decay time constants must be > 0")


# ---------------------------------------------------------------------------
# Evidence model (used by the consensus engine and gate when ConsensusConfig.evidence is set)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceConfig:
    """
    Makes the gate depend on independent evidence instead of any single physical relationship.

      families           Contradictions are grouped into families (door->PIR, light, occupancy,
                         PIR shape, DHT). A window is SUSPICIOUS on ONE strong contradiction, or on
                         weak contradictions from TWO DIFFERENT families. Light evidence is weak, so
                         it can never freeze learning on its own, however often it fails.
      auto_light         The door->light relationship is enforced only if calibration showed it
                         holds in THIS room: at least `light_reliable_min` of at least
                         `light_min_events` door openings were followed by a light step. Otherwise a
                         missing light response counts for nothing (a dark corridor, a bright room).
      temporal_bucket    Unexplained PIR seconds in a room believed EMPTY accumulate in a leaky
                         bucket (time constant `bucket_tau_s`). A contradiction is raised when it
                         exceeds max(bucket_threshold_s, bucket_background_k x the level calibration's
                         own false triggers would sustain), so a noisy PIR is not punished and sparse
                         but persistent noise is not ignored.
    """
    families: bool = True
    auto_light: bool = True
    light_reliable_min: float = 0.8
    light_min_events: int = 8
    temporal_bucket: bool = True
    bucket_tau_s: float = 600.0
    bucket_threshold_s: float = 8.0
    bucket_background_k: float = 10.0   # measured: k=4 raised benign false SUSPICIOUS 4x in a noisy-PIR room; k=10 keeps most of the protection

    def __post_init__(self) -> None:
        if not 0.0 < self.light_reliable_min <= 1.0 or self.light_min_events < 1:
            raise ValueError("light_reliable_min must be in (0, 1] and light_min_events >= 1")
        if min(self.bucket_tau_s, self.bucket_threshold_s, self.bucket_background_k) <= 0:
            raise ValueError("bucket parameters must be > 0")


# ---------------------------------------------------------------------------
# Cross-sensor consensus and the trust gate (must NOT depend on mu/Sigma)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConsensusConfig:
    """
    Consensus asks "do the sensors corroborate each other?" from FIXED physical relationships
    only, never from the learned mu/Sigma. Otherwise an attacker who has dragged the baseline
    also drags the gate meant to resist them.

    Relationships checked per window (each yields positive or negative evidence):
      door -> PIR     a confirmed door opening is followed by motion (someone passed through)
      door -> light   an opening lets corridor light in, so the LDR steps up
      PIR  <- person  motion in a room believed EMPTY, with no door opening to explain it, is
                      contradictory ("motion with no entry"). Occupancy is tracked from door
                      openings + PIR; motion at a desk while someone is inside is NORMAL.
      PIR  continuity the PIR cannot stay continuously high for minutes
      temp <-> hum    a sudden temperature step must come with a humidity response
      light steps     an abrupt light step in an empty room needs a door opening to explain it
    All numbers below are starting points; several (especially door->light) depend on the real
    installation and must be re-checked on real data.
    """

    # door -> PIR: PIR must be high within [t_door - pre, t_door + post] of the confirmed opening.
    # (A confirmed opening trails the real one by the door debounce time, so motion can start
    # slightly BEFORE the confirmation.)
    door_pir_pre_s: float = 2.0
    door_pir_post_s: float = 5.0

    # door -> light: compare median raw LDR over [t-4.0, t-1.5] with [t, t+3.0] around the opening.
    #   "off"            ignore light
    #   "positive_only"  a step counts as corroboration, its absence costs nothing (use this if
    #                    the corridor outside the real door is dark)
    #   "full"           a missing step counts as weak contradiction
    light_check: str = "full"
    door_light_step_adc: float = 150.0

    # PIR motion that no door opening (and no believed occupant) explains: contradictory only if
    # at least this many seconds of it occur in one window. (A single false trigger lasts ~3 s.)
    pir_unexplained_min_s: float = 4.0

    # A PIR continuously high for this long is physically implausible. Longest seen in normal
    # simulated data: ~94 s.
    pir_max_continuous_s: float = 300.0

    # Occupancy: after an exit-like door passage, the room is believed EMPTY once the PIR has
    # been silent this long. Shorter = faster to arm but more false alarms when someone sits still.
    vacancy_timeout_s: float = 600.0
    exit_clear_s: float = 30.0          # motion this long after a passage means someone stayed

    # temp <-> hum: only checked when a window's mean temperature OR humidity moves by at least
    # this much from the previous window (normal drift is ~0.05 C per window). The response is
    # consistent if humidity moves the OPPOSITE way by between ratio_min and ratio_max % per C.
    temp_step_c: float = 1.0
    hum_step_pct: float = 4.0
    hum_per_c_range: tuple[float, float] = (0.5, 8.0)

    # Abrupt raw LDR step (counts between consecutive samples) that needs explaining.
    abrupt_light_step_adc: float = 250.0
    abrupt_light_post_s: float = 4.0

    # Evidence weights.
    w_door_pir: float = 1.0
    w_light: float = 0.5
    w_pir_unexplained: float = 1.0
    w_pir_stuck: float = 1.0
    w_temp_hum: float = 1.0
    w_abrupt_light: float = 1.0

    # --- trust classification (the gate) -------------------------------------------------
    # SUSPICIOUS: negative evidence sums to <= -suspicious_neg. Any weaker contradiction -> UNCERTAIN.
    # No contradiction: TRUSTED if the Mahalanobis score is within the anomaly threshold, or
    # if the window is a corroborated event (positive evidence >= corroborated_min) whose score
    # is not absurd (<= corroborated_max_d). Otherwise UNCERTAIN.
    suspicious_neg: float = 1.0
    corroborated_min: float = 1.0
    corroborated_max_d: float = 15.0

    # Suspicion memory: contradictions raise a level that decays by suspicion_decay per window.
    # While the level is >= suspicion_hold, any otherwise-UNCERTAIN window that is still anomalous
    # is treated as SUSPICIOUS (so a sustained attack is not slowly learned after one detection),
    # and refreshes the level by suspicion_refresh. After suspicion_max_hold_windows of this with
    # no fresh contradiction the gate relents (UNCERTAIN again), so one false alarm can never
    # freeze learning forever. 360 windows = 3 h.
    suspicion_decay: float = 0.98
    suspicion_hold: float = 0.5
    suspicion_refresh: float = 0.05
    suspicion_cap: float = 5.0
    suspicion_max_hold_windows: int = 360

    # Occupancy model. None = the original tracker (kept as the baseline); an OccupancyConfig
    # switches to the confidence-based tracker described above.
    occupancy: Optional[OccupancyConfig] = None

    # Evidence model. None = the original rules (a sum of contradictions); an EvidenceConfig switches to families.
    evidence: Optional[EvidenceConfig] = None


# ---------------------------------------------------------------------------
# Persistence / quarantine
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SecurityConfig:
    # Consecutive SUSPICIOUS windows before raising an alert / quarantine,
    # per the "Persistence check" -> "Persistent" branch in the flow diagram.
    suspicious_streak_for_alert: int = 4

    # A quarantined sensor's features are excluded from mu/Sigma updates
    # (not the whole system) until an admin clears probation.
    probation_trusted_windows_required: int = 20


# ---------------------------------------------------------------------------
# Top-level bundle - construct this once and pass it down the pipeline.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    online_stats: OnlineStatsConfig = field(default_factory=OnlineStatsConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


DEFAULT_CONFIG = Config()
DEMO_CONFIG = Config(features=DemoFeatureConfig())