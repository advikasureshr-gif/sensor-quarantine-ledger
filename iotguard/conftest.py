"""Shared fixtures. Streams are simulated, short, and cached per test session so the suite stays fast."""
from dataclasses import replace

import pytest

from iotguard.config import DEFAULT_CONFIG, CalibrationConfig
from iotguard.experiments import comparison_systems
from iotguard.sim_sensors import Attack, SimulatorSource
from iotguard.sim_world import World


@pytest.fixture(scope="session")
def new_consensus():
    """The full system: confidence occupancy tracker + evidence model."""
    return comparison_systems()["new"]


@pytest.fixture(scope="session")
def small_cfg():
    """A 1 hour calibration so a few simulated hours are enough to reach operation."""
    cal = CalibrationConfig(min_duration_s=3600.0, min_trusted_windows=40, max_duration_s=14400.0, check_every_windows=20)
    return replace(DEFAULT_CONFIG, calibration=cal)


@pytest.fixture(scope="session")
def stream():
    """12 simulated hours (08:00-20:00), benign."""
    return list(SimulatorSource(World(seed=4), 12 * 3600))


@pytest.fixture(scope="session")
def attack_stream():
    """A PIR stuck high for 5 hours starting at t = 3 h (the room is calibrated by then), 10 h in all."""
    return list(SimulatorSource(World(seed=4), 10 * 3600, attacks=[Attack("force", "pir", 3 * 3600, 8 * 3600, magnitude=1, tag="pir_stuck_high")]))


def feed(pipe, readings):
    out = []
    for r in readings:
        out += pipe.push(r)
    return out


def key(outcomes):
    """A comparable summary of outcomes."""
    return [(o.index, o.phase.value, None if o.decision is None else o.decision.label.value,
             None if o.score != o.score else round(o.score, 6)) for o in outcomes]