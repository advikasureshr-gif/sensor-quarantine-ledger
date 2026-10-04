"""
experiments.py - run scenarios many times and measure them properly.

Two stages, because only the first is expensive:

  prepare(spec)     simulate -> preprocess -> windows -> consensus. Slow (~5 s per simulated
                    day), cached on disk, independent of the learned model.
  replay(prepared, params)
                    learn from the prepared windows with a given model/gate setting, next to
                    a naive always-learn twin. Takes about a second, so learning rates,
                    thresholds and gate settings can be swept freely without re-simulating.

Measuring poisoning properly
----------------------------
A model drifts even on benign data (weekends, weather), so total drift proves nothing. Every
attacked run is therefore compared with a MATCHED benign run: same seed, same length, same
settings, just no attack. The reported effect is the EXCESS drift (attacked minus benign), and
"protection" is the naive model's excess minus the gated model's excess. Results are
aggregated over seeds with 95% confidence intervals (Student t on the per-seed differences),
so a difference that is not distinguishable from zero is reported as such.

Caching: results are keyed on everything that influences them (scenario, seed, length,
strength, consensus/world/sensor/preprocess/feature settings). If you change the LOGIC of the
preprocessing, features or consensus code, bump CODE_VERSION so stale caches are not reused.

Command line
------------
    python -m iotguard.experiments excess --scenario gradual_poison --seeds 0-4 \\
           --strengths 0.5,1,2 --workers 4 --cache runs_cache [--budget 270]
Runs resume from the cache, so a --budget (seconds) just stops early and the next call continues.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import (DEFAULT_CONFIG, ConsensusConfig, OnlineStatsConfig)
from .consensus import ConsensusEngine, ConsensusResult, Gate, Trust
from .features import extract_windows
from .gaussian import GaussianModel
from .preprocess import clean
from .scenarios import ScenarioSource
from .sim_sensors import SensorConfig
from .sim_world import WorldConfig
from .sources import strip_labels

CODE_VERSION = "1"          # bump when preprocessing / feature / consensus LOGIC changes
TRUTH = {"normal": 0, "legit": 1, "attack": 2}
LABELS = (Trust.TRUSTED, Trust.UNCERTAIN, Trust.SUSPICIOUS)


# ---------------------------------------------------------------------------
# a run specification and its prepared form
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    scenario: str
    seed: int
    days: Optional[float] = None            # None = the scenario's default length
    strength: float = 1.0
    cons: ConsensusConfig = ConsensusConfig()
    world: WorldConfig = WorldConfig()
    sensors: SensorConfig = SensorConfig()

    def benign(self) -> "RunSpec":
        """The matched benign run: same seed, length and settings, no attack."""
        return replace(self, scenario="normal", strength=1.0)

    def key(self) -> str:
        fp = repr((CODE_VERSION, self.scenario, self.seed, self.days, self.strength, self.cons, self.world,
                   self.sensors, DEFAULT_CONFIG.preprocess, DEFAULT_CONFIG.features))
        return f"{self.scenario}_s{self.seed}_{hashlib.sha1(fp.encode()).hexdigest()[:12]}"


@dataclass
class Prepared:
    spec: RunSpec
    names: tuple[str, ...]
    calib_s: float
    duration_s: float
    X_cal: np.ndarray                       # calibration windows (valid only)
    t: np.ndarray                           # operation windows: start times
    X: np.ndarray
    cons: list[ConsensusResult]
    truth: np.ndarray                       # int8: 0 normal, 1 legitimate change, 2 attack
    attack_iv: list[tuple[float, float]] = field(default_factory=list)
    legit_iv: list[tuple[float, float]] = field(default_factory=list)


def _path(cache_dir, spec: RunSpec) -> Optional[Path]:
    return None if cache_dir is None else Path(cache_dir) / (spec.key() + ".pkl")


def prepare(spec: RunSpec, cache_dir=None) -> Prepared:
    """Simulate and analyse one run (cached). Slow the first time."""
    path = _path(cache_dir, spec)
    if path is not None and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    src = ScenarioSource(spec.scenario, spec.seed, days=spec.days, world_cfg=spec.world,
                         sensors=spec.sensors, strength=spec.strength)
    b = src.built
    eng = ConsensusEngine(spec.cons)
    cal, ts, xs, cons, truth = [], [], [], [], []
    for w in extract_windows(clean(strip_labels(src)), keep_samples=True):
        if not w.valid:
            continue
        c = eng.evaluate(w)
        if w.t_end <= src.calibration_s:
            cal.append(w.x)
            continue
        atk = any(a.start_s < w.t_end and a.end_s > w.t_start for a in b.attacks)
        leg = any(ch.start_s < w.t_end and ch.label_end_s > w.t_start for ch in b.changes)
        ts.append(w.t_start); xs.append(w.x.copy()); cons.append(c)
        truth.append(TRUTH["attack"] if atk else (TRUTH["legit"] if leg else TRUTH["normal"]))
    prep = Prepared(spec, DEFAULT_CONFIG.features.feature_names, src.calibration_s, src.duration_s,
                    np.vstack(cal), np.array(ts), np.vstack(xs), cons, np.array(truth, dtype=np.int8),
                    [(a.start_s, a.end_s) for a in b.attacks],
                    [(ch.start_s, ch.label_end_s) for ch in b.changes])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(prep, f)
        tmp.replace(path)
    return prep


def _worker(args):
    spec, cache_dir = args
    prepare(spec, cache_dir)
    return spec.key()


def prepare_many(specs: Sequence[RunSpec], cache_dir, workers: int = 1, budget_s: Optional[float] = None,
                 verbose: bool = True) -> dict[RunSpec, Optional[Prepared]]:
    """
    Prepare every spec, in parallel if workers > 1, resuming from the cache. When `budget_s` runs
    out, stops starting new work; call again to continue. Returns {spec: Prepared or None if not done}.
    """
    unique = list(dict.fromkeys(specs))
    todo = [s for s in unique if not (_path(cache_dir, s) and _path(cache_dir, s).exists())]
    t0 = time.time()
    if verbose and todo:
        print(f"preparing {len(todo)} of {len(unique)} runs ({workers} worker(s))...", flush=True)
    if workers <= 1:
        for s in todo:
            if budget_s is not None and time.time() - t0 > budget_s:
                break
            prepare(s, cache_dir)
            if verbose:
                print(f"  done {s.key()}  ({time.time() - t0:.0f}s)", flush=True)
    elif todo:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            it = pool.imap_unordered(_worker, [(s, cache_dir) for s in todo])
            for _ in todo:
                try:
                    k = it.next(timeout=None if budget_s is None else max(1.0, budget_s - (time.time() - t0)))
                except mp.TimeoutError:
                    pool.terminate()
                    break
                if verbose:
                    print(f"  done {k}  ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for s in unique:
        p = _path(cache_dir, s)
        out[s] = prepare(s, cache_dir) if (p and p.exists()) else None
    return out


# ---------------------------------------------------------------------------
# replaying with a model setting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayParams:
    online: OnlineStatsConfig = DEFAULT_CONFIG.online_stats
    gate: ConsensusConfig = ConsensusConfig()
    d_threshold: float = DEFAULT_CONFIG.anomaly.mahalanobis_threshold
    naive_alpha: Optional[float] = None         # None = same as online.alpha_trusted


@dataclass
class Replay:
    prep: Prepared
    params: ReplayParams
    D: np.ndarray                               # gated model's score for each window (before learning from it)
    Dn: np.ndarray                              # naive twin's score
    label: np.ndarray                           # int8: 0 TRUSTED, 1 UNCERTAIN, 2 SUSPICIOUS
    drift: np.ndarray                           # gated model's baseline drift distance after each window
    drift_n: np.ndarray                         # naive twin's

    @property
    def t(self) -> np.ndarray:
        return self.prep.t


def replay(prep: Prepared, params: ReplayParams = ReplayParams()) -> Replay:
    ocfg = params.online
    model = GaussianModel(ocfg, prep.names)
    model.fit(prep.X_cal)
    naive_alpha = ocfg.alpha_trusted if params.naive_alpha is None else params.naive_alpha
    naive = GaussianModel(replace(ocfg, drift_cap_std=None), prep.names)
    naive.fit(prep.X_cal)
    gate = Gate(params.gate, params.d_threshold)
    n = len(prep.t)
    D, Dn, lab = np.empty(n), np.empty(n), np.empty(n, dtype=np.int8)
    dr, drn = np.empty(n), np.empty(n)
    index = {l: i for i, l in enumerate(LABELS)}
    for i in range(n):
        x = prep.X[i]
        D[i], Dn[i] = model.score(x), naive.score(x)
        dec = gate.decide(prep.cons[i], D[i])
        model.update(x, dec.alpha(ocfg))
        naive.update(x, naive_alpha)
        lab[i] = index[dec.label]
        dr[i], drn[i] = model.drift_distance(), naive.drift_distance()
    return Replay(prep, params, D, Dn, lab, dr, drn)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _last_day(t: np.ndarray) -> np.ndarray:
    return t > t.max() - 86400.0


def metrics(rep: Replay) -> dict:
    """Detection, false-alarm and drift numbers for one replayed run."""
    p, t, lab, tr = rep.prep, rep.t, rep.label, rep.prep.truth
    atk, nonatk = tr == TRUTH["attack"], tr != TRUTH["attack"]
    th = rep.params.d_threshold
    last = _last_day(t)
    out = {
        "n_windows": len(t),
        "frac_trusted": float((lab == 0).mean()), "frac_uncertain": float((lab == 1).mean()),
        "frac_suspicious": float((lab == 2).mean()),
        "false_susp": float((lab[nonatk] == 2).mean()) if nonatk.any() else math.nan,
        "false_susp_benign": float((lab[tr == TRUTH["normal"]] == 2).mean()) if (tr == TRUTH["normal"]).any() else math.nan,
        "drift_last24": float(rep.drift[last].mean()), "drift_naive_last24": float(rep.drift_n[last].mean()),
        "flag_last24": float((rep.D[last] > th).mean()), "flag_naive_last24": float((rep.Dn[last] > th).mean()),
    }
    if atk.any():
        start = min(a for a, _ in p.attack_iv)
        hit = np.flatnonzero(atk & (lab == 2))
        out["detected"] = bool(len(hit))
        out["latency_min"] = float((t[hit[0]] - start) / 60.0) if len(hit) else math.nan
        out["frozen_attack"] = float((lab[atk] == 2).mean())
        out["n_attack"] = int(atk.sum())
    return out


def excess(attacked: Replay, benign: Replay) -> dict:
    """Excess drift = attacked run minus its matched benign run (same seed, length and settings)."""
    if len(attacked.t) != len(benign.t) or not np.array_equal(attacked.t, benign.t):
        raise ValueError("runs are not matched: window timelines differ")
    last = _last_day(attacked.t)
    g = float(attacked.drift[last].mean() - benign.drift[last].mean())
    n = float(attacked.drift_n[last].mean() - benign.drift_n[last].mean())
    return {"excess_gated": g, "excess_naive": n, "protection": n - g,
            "benign_drift": float(benign.drift[last].mean()), "benign_drift_naive": float(benign.drift_n[last].mean())}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Agg:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero."""
        return self.n > 1 and (self.lo > 0 or self.hi < 0)

    def __str__(self) -> str:
        if self.n == 0 or math.isnan(self.mean):
            return "n/a"
        if self.n == 1 or math.isnan(self.lo):
            return f"{self.mean:+.2f} (n=1)"
        return f"{self.mean:+.2f} [{self.lo:+.2f}, {self.hi:+.2f}]"


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def _tcrit(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        if df in _T975:
            return _T975[df]
        lower = [d for d in _T975 if d <= df]
        return _T975[max(lower)] if df < 30 else 1.96


def aggregate(values: Iterable[float]) -> Agg:
    """Mean and 95% Student-t interval of per-seed values (NaNs dropped)."""
    v = np.array([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return Agg(math.nan, math.nan, math.nan, 0)
    if n == 1:
        return Agg(float(v[0]), math.nan, math.nan, 1)
    half = _tcrit(n - 1) * v.std(ddof=1) / math.sqrt(n)
    return Agg(float(v.mean()), float(v.mean() - half), float(v.mean() + half), n)


# ---------------------------------------------------------------------------
# the excess-drift experiment
# ---------------------------------------------------------------------------
def excess_experiment(scenario: str, seeds: Sequence[int], strengths: Sequence[float], cache_dir,
                      params: ReplayParams = ReplayParams(), days: Optional[float] = None, workers: int = 1,
                      budget_s: Optional[float] = None, verbose: bool = True,
                      cons: ConsensusConfig = ConsensusConfig(),
                      world: WorldConfig = WorldConfig(), sensors: SensorConfig = SensorConfig()) -> dict:
    """
    For each strength: per-seed excess drift against a matched benign run, aggregated over seeds.
    Returns {"rows": [...per seed...], "summary": {strength: {...Agg...}}, "missing": n_runs_not_done}.
    """
    base = dict(days=days, cons=cons, world=world, sensors=sensors)
    specs = [RunSpec(scenario, s, strength=st, **base) for s in seeds for st in strengths]
    specs += [RunSpec(scenario, s, **base).benign() for s in seeds]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    missing = sum(1 for p in prepared.values() if p is None)
    rows, summary = [], {}
    benign_cache: dict[int, Replay] = {}
    for s in seeds:
        bs = RunSpec(scenario, s, **base).benign()
        if prepared.get(bs) is not None:
            benign_cache[s] = replay(prepared[bs], params)
    for st in strengths:
        per_seed = []
        for s in seeds:
            spec = RunSpec(scenario, s, strength=st, **base)
            if prepared.get(spec) is None or s not in benign_cache:
                continue
            rep = replay(prepared[spec], params)
            row = {"scenario": scenario, "seed": s, "strength": st, **metrics(rep), **excess(rep, benign_cache[s])}
            rows.append(row); per_seed.append(row)
        keys = ("excess_gated", "excess_naive", "protection", "false_susp", "frozen_attack", "latency_min")
        summary[st] = {k: aggregate(r.get(k, math.nan) for r in per_seed) for k in keys}
        summary[st]["detected"] = (sum(1 for r in per_seed if r.get("detected")), len(per_seed))
        summary[st]["benign_drift"] = aggregate(r["benign_drift"] for r in per_seed)
    return {"rows": rows, "summary": summary, "missing": missing}


def format_excess(result: dict, scenario: str) -> str:
    lines = [f"{scenario}: excess baseline drift versus a matched benign run (mean [95% CI] over seeds; "
             f"'*' = interval excludes 0)"]
    lines.append(f"{'strength':>9} {'seeds':>6} | {'excess drift GATED':>24} {'excess drift NAIVE':>24} {'protection (naive-gated)':>26} | "
                 f"{'detected':>9} {'latency min':>12} {'attack frozen':>14} {'false SUSP':>11}")
    for st, s in result["summary"].items():
        n = s["excess_gated"].n
        if n == 0:
            continue
        mark = lambda a: str(a) + ("*" if a.significant else " ")
        det = s["detected"]
        lat = s["latency_min"]
        lines.append(f"{st:>9g} {n:>6} | {mark(s['excess_gated']):>24} {mark(s['excess_naive']):>24} {mark(s['protection']):>26} | "
                     f"{f'{det[0]}/{det[1]}':>9} {('%.0f' % lat.mean) if lat.n else 'never':>12} "
                     f"{s['frozen_attack'].mean:>14.0%} {s['false_susp'].mean:>11.2%}")
    b = next((s["benign_drift"] for s in result["summary"].values() if s["benign_drift"].n), None)
    if b is not None:
        lines.append(f"(reference: the benign runs' own drift, last 24 h, gated model: {b.mean:.2f})")
    if result["missing"]:
        lines.append(f"[{result['missing']} run(s) not finished yet; run again to continue from the cache]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m iotguard.experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("excess", help="excess drift vs matched benign runs, over seeds and strengths")
    ex.add_argument("--scenario", required=True)
    ex.add_argument("--seeds", default="0-4")
    ex.add_argument("--strengths", default="0.5,1,2")
    ex.add_argument("--days", type=float, default=None)
    ex.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 0))
    ex.add_argument("--cache", default="runs_cache")
    ex.add_argument("--budget", type=float, default=None, help="stop starting new runs after this many seconds")
    a = ap.parse_args(argv)
    res = excess_experiment(a.scenario, _parse_seeds(a.seeds), [float(x) for x in a.strengths.split(",")], a.cache,
                            days=a.days, workers=a.workers, budget_s=a.budget)
    print(format_excess(res, a.scenario))
    return 0


if __name__ == "__main__":
    sys.exit(_main())