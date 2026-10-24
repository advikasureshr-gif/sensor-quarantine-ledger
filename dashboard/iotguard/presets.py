"""
presets.py - ready-made configurations.

room_setup()        the settings validated on the simulator: the full new system (confidence occupancy tracker with a 12 h
                    horizon, evidence model, screened calibration over 2 days). Meant for a real room over days.

demo_setup(factor)  the SAME system with every long timer divided by `factor`, for a tabletop demonstration where you
                    cannot wait hours: calibration (2 days -> 2 days/factor), the occupancy tracker's vacancy, horizon and
                    decay times, the PIR-continuity limit and the temporal bucket.
                    NOT scaled: the window length (30 s), the per-window thresholds, the physical delays (how soon motion
                    follows a door opening) and the quarantine episode length (2 h). Compressing the last one made the PIR
                    get quarantined by ordinary false alarms in a benign simulation, because false-alarm episodes last up to
                    about 45 minutes whatever the demo's pace; a tabletop demo of quarantine needs a deliberately long attack.

                    WARNING: demo_setup has not been validated, on the simulator or on hardware. The simulator's room
                    follows a real daily routine and cannot be compressed, so there is nothing to test a compressed
                    timeline against. Treat it as a convenience for showing the mechanisms, and check what it does on your
                    own recordings before relying on it.
"""
from __future__ import annotations

from dataclasses import replace

from .config import (DEFAULT_CONFIG, CalibrationConfig, Config, ConsensusConfig, EvidenceConfig, OccupancyConfig, SecurityConfig)


def room_setup(horizon_h: float = 12.0) -> tuple[Config, ConsensusConfig]:
    return DEFAULT_CONFIG, ConsensusConfig(occupancy=OccupancyConfig(pir_horizon_s=horizon_h * 3600.0), evidence=EvidenceConfig())


def demo_setup(factor: float = 12.0, horizon_h: float = 12.0) -> tuple[Config, ConsensusConfig]:
    if factor < 1.0:
        raise ValueError("factor must be >= 1")
    cfg, cons = room_setup(horizon_h)
    o, e, c = cons.occupancy, cons.evidence, cfg.calibration
    cons = replace(
        cons,
        vacancy_timeout_s=cons.vacancy_timeout_s / factor, pir_max_continuous_s=max(60.0, cons.pir_max_continuous_s / factor),
        occupancy=replace(o, pir_horizon_s=o.pir_horizon_s / factor, exit_decay_tau_s=o.exit_decay_tau_s / factor,
                          expired_decay_tau_s=o.expired_decay_tau_s / factor),
        evidence=replace(e, bucket_tau_s=max(60.0, e.bucket_tau_s / factor)),
    )
    cal = replace(c, min_duration_s=c.min_duration_s / factor, max_duration_s=c.max_duration_s / factor,
                  min_trusted_windows=max(30, int(c.min_trusted_windows / factor)))
    return replace(cfg, calibration=cal), cons