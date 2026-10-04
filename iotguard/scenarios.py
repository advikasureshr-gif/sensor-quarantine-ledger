"""
scenarios.py - named experiment presets.

A scenario = a legitimate-change schedule + an attack schedule + where calibration ends.
Every scenario has the same shape, so results are comparable:

    day 0-1   CALIBRATION   (2 days, clean, unless the scenario is a calibration attack)
    day 2-6   OPERATION     (5 days; changes and attacks happen here, from day 3 on)

Clock times are written as at(day, hour) = "day N at HH:00", resolved against the world's
start_hour, so the same scenario stays correct if you change when the run starts.
Default calendar: day 0 = Monday 08:00, so operation starts Wednesday 08:00 and
days 5-6 are the (mostly empty) weekend.

Usage
-----
    from iotguard.scenarios import make_source, list_scenarios
    src = make_source("gradual_poison", seed=42)
    src.calibration_s          # where calibration ends (sim seconds)
    for reading in strip_labels(src): ...   # pipeline never sees labels

    python -m iotguard.scenarios list
    python -m iotguard.scenarios record --out logs --seed 0 --days 7 normal gradual_poison

Which project experiment each scenario serves is in `Scenario.experiment`; what a good
system should do (including where we EXPECT the method to be weak) is in `Scenario.expect`.
Those expectations are hypotheses to test, not results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from .sim_sensors import Attack, SensorConfig, SimulatorSource
from .sim_world import DAY_S, LegitChange, World, WorldConfig
from .sources import write_jsonl

At = Callable[[int, float], float]
Factory = Callable[[At], tuple[Sequence[LegitChange], Sequence[Attack]]]


# ---------------------------------------------------------------------------
# scenario definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    experiment: str                     # which project experiment this serves
    expect: str                         # what a good system should do (a hypothesis)
    factory: Factory
    calibration_days: float = 2.0
    days: float = 7.0
    contaminates_calibration: bool = False

    def build(self, world_cfg: WorldConfig = WorldConfig(), days: Optional[float] = None,
              strength: float = 1.0) -> "BuiltScenario":
        if strength <= 0:
            raise ValueError("strength must be > 0")
        start_h = world_cfg.start_hour
        at: At = lambda day, hour: day * DAY_S + (hour - start_h) * 3600.0
        changes, attacks = self.factory(at)
        changes, attacks = _scaled(changes, attacks, strength)
        total = (self.days if days is None else days) * DAY_S
        calib = self.calibration_days * DAY_S
        if calib >= total:
            raise ValueError(f"{self.name}: duration ({total/DAY_S:g} d) must exceed calibration ({self.calibration_days:g} d)")
        for ch in changes:
            if ch.start_s < calib:
                raise ValueError(f"{self.name}: legitimate change starts during calibration")
        for a in attacks:
            if a.start_s < calib and not self.contaminates_calibration:
                raise ValueError(f"{self.name}: attack starts during calibration but the scenario is not a calibration attack")
            if a.start_s >= total:
                raise ValueError(f"{self.name}: attack {a.tag!r} starts after the run ends; increase days")
        return BuiltScenario(self, calib, total, tuple(changes), tuple(attacks), world_cfg, strength)


@dataclass(frozen=True)
class BuiltScenario:
    scenario: Scenario
    calibration_s: float
    duration_s: float
    changes: tuple[LegitChange, ...]
    attacks: tuple[Attack, ...]
    world_cfg: WorldConfig
    strength: float = 1.0


def _scaled(changes, attacks, strength: float):
    """
    Scale how strong a scenario is. offset attacks: size x strength. force/replay attacks: the
    fraction of samples affected (peak) x strength, capped at 1 (so 'stuck' attacks become
    intermittent below 1). Legitimate changes: magnitude x strength. Timing never changes.
    """
    if strength == 1.0:
        return list(changes), list(attacks)
    atk = [replace(a, magnitude=a.magnitude * strength) if a.kind == "offset"
           else replace(a, peak=min(1.0, a.peak * strength)) for a in attacks]
    return [replace(c, magnitude=c.magnitude * strength) for c in changes], atk


# ---------------------------------------------------------------------------
# the scenarios
# ---------------------------------------------------------------------------
def _normal(at):
    return [], []


def _legit_brighter(at):          # blinds removed: 50% more daylight, from Thursday 07:00
    return [LegitChange("brighter", at(3, 7), 0.5)], []


def _legit_warmer(at):            # room runs 4 C warmer from Thursday 07:00 (season/AC change)
    return [LegitChange("warmer", at(3, 7), 4.0, ramp_s=3600)], []


def _legit_busier(at):            # twice the activity and visitors from Thursday 07:00
    return [LegitChange("busier", at(3, 7), 1.0)], []


def _legit_shifted(at):           # the occupant's routine moves 2 h later from Thursday
    return [LegitChange("shifted", at(3, 0), 2.0)], []


def _pir_stuck_night(at):         # Thu 22:00-02:00, room empty
    return [], [Attack("force", "pir", at(3, 22), at(4, 2), magnitude=1, tag="pir_stuck_high")]


def _pir_stuck_day(at):           # Thu 10:00-12:00, occupant normally inside
    return [], [Attack("force", "pir", at(3, 10), at(3, 12), magnitude=1, tag="pir_stuck_high")]


def _door_masked(at):             # door reported closed all Thursday working day
    return [], [Attack("force", "door", at(3, 7), at(3, 19), magnitude=0, tag="door_masked")]


def _ldr_blind(at):               # LDR reads dark Thursday 13:00-18:00 (covered / spoofed)
    return [], [Attack("force", "ldr", at(3, 13), at(3, 18), magnitude=60, tag="ldr_blind")]


def _temp_spike(at):              # +8 C on the DHT, Thursday 10:00-12:00
    return [], [Attack("offset", "temp", at(3, 10), at(3, 12), magnitude=8.0, tag="temp_spike")]


def _sudden_poison(at):           # several channels abnormal at once for 6 h, Thursday daytime
    s, e = at(3, 10), at(3, 16)
    return [], [
        Attack("offset", "temp", s, e, magnitude=6.0, tag="sudden_poison"),
        Attack("offset", "hum", s, e, magnitude=-15.0, tag="sudden_poison"),
        Attack("offset", "ldr", s, e, magnitude=500, tag="sudden_poison"),
        Attack("force", "pir", s, e, magnitude=1, peak=0.4, tag="sudden_poison"),
    ]


def _gradual_poison(at):          # the key experiment: slow multi-sensor drift, then held
    s, ramp = at(3, 8), 36 * 3600.0
    return [], [
        Attack("offset", "temp", s, math.inf, magnitude=5.0, ramp_s=ramp, tag="gradual_poison"),
        Attack("offset", "hum", s, math.inf, magnitude=-12.0, ramp_s=ramp, tag="gradual_poison"),
        Attack("offset", "ldr", s, math.inf, magnitude=400, ramp_s=ramp, tag="gradual_poison"),
        Attack("force", "pir", s, math.inf, magnitude=1, ramp_s=ramp, peak=0.3, tag="gradual_poison"),
    ]


def _gradual_dht(at):             # DHT-only drift: no other sensor corroborates temperature
    s, ramp = at(3, 8), 48 * 3600.0
    return [], [
        Attack("offset", "temp", s, math.inf, magnitude=5.0, ramp_s=ramp, tag="gradual_dht"),
        Attack("offset", "hum", s, math.inf, magnitude=-12.0, ramp_s=ramp, tag="gradual_dht"),
    ]


def _calibration_attack(at):      # teach the baseline that night-time PIR activity is "normal"...
    pattern = dict(kind="force", channel="pir", magnitude=1, peak=0.3, tag="night_pir_pattern")
    return [], [
        Attack(start_s=at(0, 22), end_s=at(1, 3), **pattern),
        Attack(start_s=at(1, 22), end_s=at(2, 3), **pattern),
        Attack(start_s=at(3, 22), end_s=at(4, 3), **pattern),   # ...then exploit it in operation
    ]


def _persistent_pir_night(at):    # 12 h of PIR stuck high overnight: alert / quarantine test
    return [], [Attack("force", "pir", at(3, 20), at(4, 8), magnitude=1, tag="pir_stuck_high")]


def _mixed_legit_attack(at):      # adapt to a legitimate change, still catch attacks afterwards
    return ([LegitChange("brighter", at(3, 7), 0.5)],
            [Attack("offset", "temp", at(4, 10), at(4, 12), magnitude=8.0, tag="temp_spike"),
             Attack("force", "pir", at(4, 22), at(5, 2), magnitude=1, tag="pir_stuck_high")])


def _replay_pir_night(at):        # night-time PIR replays the activity of 12 h earlier
    return [], [Attack("replay", "pir", at(3, 22), at(4, 2), lag_s=12 * 3600.0, tag="pir_replay")]


_LIST = [
    Scenario("normal", "Normal behaviour only", "Exp 1",
             "Learns the routine; few false alarms; baseline stays put.", _normal),
    Scenario("legit_brighter", "Legitimate change: room gets brighter", "Exp 2 / 8",
             "Adapts to the new light level instead of treating it as an attack.", _legit_brighter),
    Scenario("legit_warmer", "Legitimate change: room runs warmer", "Exp 2 / 8",
             "Follows the slow temperature shift. Watch the drift cap: it must not block real change forever.", _legit_warmer),
    Scenario("legit_busier", "Legitimate change: busier room", "Exp 2 / 8",
             "Adapts to doubled activity and visitors.", _legit_busier),
    Scenario("legit_shifted", "Legitimate change: routine moves 2 h later", "Exp 2 / 8",
             "Adapts to the shifted schedule after a transition period.", _legit_shifted),
    Scenario("pir_stuck_night", "PIR stuck high overnight (room empty)", "Exp 3",
             "Detectable: motion with no entry. Expect SUSPICIOUS and a frozen baseline.", _pir_stuck_night),
    Scenario("pir_stuck_day", "PIR stuck high while the occupant is inside", "Exp 3",
             "Hard case: PIR high is plausible when someone is inside. Reports a limit of consensus.", _pir_stuck_day),
    Scenario("door_masked", "Door reported closed all working day", "Exp 3",
             "Detectable indirectly: motion and light changes with no door events (entries go unseen).", _door_masked),
    Scenario("ldr_blind", "LDR reads dark while room is lit", "Exp 3",
             "Partly detectable: lamp and corridor-light steps vanish while occupancy is known.", _ldr_blind),
    Scenario("temp_spike", "DHT temperature +8 C for 2 h", "Exp 3 / 4",
             "Caught by Mahalanobis; humidity's physical link to temperature adds weak corroboration.", _temp_spike),
    Scenario("sudden_poison", "Sudden multi-channel poisoning, 6 h", "Exp 4",
             "Freeze quickly. The always-update baseline absorbs the shift.", _sudden_poison),
    Scenario("gradual_poison", "Gradual multi-sensor poisoning (36 h ramp, then held)", "Exp 5",
             "THE key experiment: naive baseline drifts toward the attack, gated model should resist. Plot baseline drift.", _gradual_poison),
    Scenario("gradual_dht", "Gradual DHT-only drift (48 h ramp)", "Exp 5",
             "Expected WEAKNESS: no sensor corroborates the DHT, so only the drift cap helps. Report honestly.", _gradual_dht),
    Scenario("calibration_attack", "Night PIR pattern taught during calibration, reused later", "Exp 6",
             "Calibration screening should keep the pattern out of the baseline; naive baseline accepts it as normal.", _calibration_attack,
             contaminates_calibration=True),
    Scenario("persistent_pir_night", "PIR stuck high for 12 h overnight", "Exp 7",
             "Persistent SUSPICIOUS -> alert and quarantine of the PIR while other sensors keep working.", _persistent_pir_night),
    Scenario("mixed_legit_attack", "Legitimate change, then two attacks", "Exp 8",
             "Adapt to the brighter room, still catch the later attacks (must not become over-conservative).", _mixed_legit_attack),
    Scenario("replay_pir_night", "Night PIR replays activity from 12 h earlier", "Exp 3 / 7",
             "Replay looks like real activity; only the missing entry event betrays it.", _replay_pir_night),
]
SCENARIOS: dict[str, Scenario] = {s.name: s for s in _LIST}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; available: {', '.join(SCENARIOS)}") from None


def list_scenarios() -> str:
    w = max(len(n) for n in SCENARIOS)
    lines = [f"{'name':<{w}}  {'exp':<9} title", f"{'-'*w}  {'-'*9} {'-'*40}"]
    lines += [f"{s.name:<{w}}  {s.experiment:<9} {s.title}" for s in SCENARIOS.values()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------
class ScenarioSource(SimulatorSource):
    """A SimulatorSource built from a named scenario. Carries calibration_s and a manifest."""

    def __init__(self, scenario: Union[str, Scenario], seed: int = 0, *, days: Optional[float] = None,
                 world_cfg: WorldConfig = WorldConfig(), sensors: SensorConfig = SensorConfig(),
                 strength: float = 1.0) -> None:
        sc = get_scenario(scenario) if isinstance(scenario, str) else scenario
        built = sc.build(world_cfg, days, strength)
        super().__init__(World(seed, world_cfg, built.changes), built.duration_s, sensors, built.attacks, seed=seed)
        self.scenario, self.built, self.calibration_s = sc, built, built.calibration_s

    def manifest(self) -> dict:
        """Everything an evaluator needs to score a run, as plain JSON-able data."""
        b = self.built
        inf = lambda x: None if math.isinf(x) else x
        return {
            "scenario": self.scenario.name, "title": self.scenario.title,
            "experiment": self.scenario.experiment, "expect": self.scenario.expect,
            "seed": self.seed, "strength": b.strength, "sample_period_s": b.world_cfg.sample_period_s,
            "start_hour": b.world_cfg.start_hour, "start_day_of_week": b.world_cfg.start_day_of_week,
            "calibration_s": b.calibration_s, "duration_s": b.duration_s,
            "contaminates_calibration": self.scenario.contaminates_calibration,
            "attacks": [{"tag": a.tag, "kind": a.kind, "channel": a.channel, "start_s": a.start_s,
                         "end_s": inf(a.end_s), "magnitude": a.magnitude, "ramp_s": a.ramp_s,
                         "peak": a.peak, "lag_s": a.lag_s} for a in b.attacks],
            "legit_changes": [{"kind": c.kind, "start_s": c.start_s, "label_end_s": c.label_end_s,
                               "magnitude": c.magnitude} for c in b.changes],
        }


def make_source(name: Union[str, Scenario], seed: int = 0, **kw) -> ScenarioSource:
    return ScenarioSource(name, seed, **kw)


# ---------------------------------------------------------------------------
# command line:  list | record
# ---------------------------------------------------------------------------
def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m iotguard.scenarios", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show all scenarios")
    rec = sub.add_parser("record", help="write labelled .jsonl logs (+ .json manifest) to disk")
    rec.add_argument("names", nargs="+", help="scenario names, or 'all'")
    rec.add_argument("--out", default="logs")
    rec.add_argument("--seed", type=int, default=0)
    rec.add_argument("--days", type=float, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print(list_scenarios())
        return 0
    names = list(SCENARIOS) if args.names == ["all"] else args.names
    out = Path(args.out)
    for name in names:
        src = make_source(name, args.seed, days=args.days)
        stem = f"{name}_seed{args.seed}"
        n = write_jsonl(src, out / f"{stem}.jsonl")
        (out / f"{stem}.json").write_text(json.dumps(src.manifest(), indent=2))
        mb = (out / f"{stem}.jsonl").stat().st_size / 1e6
        print(f"{stem}: {n:,} readings, {mb:.0f} MB -> {out / (stem + '.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())