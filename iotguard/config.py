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
    # windows, alpha 0.05 is a time constant of ~20 windows (10 minutes), alpha 0.01 ~50 minutes.
    #   d = x - mu;   mu += alpha * d;   Sigma = (1 - alpha) * (Sigma + alpha * d d^T)
    # These are placeholders to be tuned in the experiments (they set how fast the baseline can
    # follow, and therefore also how fast an unguarded baseline can be dragged).
    alpha_trusted: float = 0.05
    alpha_uncertain: float = 0.01
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
# Cross-sensor consensus (must NOT depend on mu/Sigma - see note below)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConsensusConfig:
    """
    Consensus answers "do the sensors corroborate each other", independent of
    whether the learned baseline currently thinks the window is anomalous.
    It must be computed from fixed physical relationships only (not from
    mu/Sigma), or an attacker who has already dragged the baseline also
    drags the gate that is supposed to resist them.
    """

    # A door event is corroborated if PIR motion also occurred within this
    # many seconds of the door state change (either side).
    door_pir_corroboration_window_s: float = 5.0

    # LDR is considered to have "changed" within a window if it moved by at
    # least this many raw ADC counts from the window's starting value.
    ldr_change_threshold_adc: int = 150

    # Scoring: start at 0, add 1 for each corroborating relationship that
    # holds, subtract 1 for each that is violated (e.g. PIR fires while door
    # stayed closed AND light stayed flat). agreement_score is then compared
    # against the two thresholds below.
    trusted_min_score: int = 1          # score >= this AND mahalanobis below threshold -> TRUSTED
    suspicious_max_score: int = -1      # score <= this (regardless of mahalanobis) -> SUSPICIOUS
    # anything in between -> UNCERTAIN


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