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
from .preprocess import CleanReading, Preprocessor, clean
from .scenarios import ScenarioSource, get_scenario
from .sim_sensors import SensorConfig
from .sim_world import DAY_S, World, WorldConfig, _paint
from .sources import strip_labels

CODE_VERSION = "1"          # bump when preprocessing / feature / consensus LOGIC changes
REPLAY_VERSION = "3"        # bump when the replay/model code or the arrays it stores change (3: the gate follows each run's own settings)
TRUTH = {"normal": 0, "legit": 1, "attack": 2}
LABELS = (Trust.TRUSTED, Trust.UNCERTAIN, Trust.SUSPICIOUS)


# ---------------------------------------------------------------------------
# a run specification and its prepared form
# ---------------------------------------------------------------------------
def _compat(fp: str) -> str:
    """
    Drop options that are switched OFF from a fingerprint, so runs prepared before those options existed
    stay valid: the occupancy and evidence models (None) and the simulator's abrupt-light events (0 per hour).
    """
    return (fp.replace(", occupancy=None, evidence=None)", ")").replace(", evidence=None)", ")")
              .replace(", cloud_events_per_h=0.0, cloud_drop_range=(0.3, 0.7), cloud_duration_range_s=(60.0, 600.0))", ")"))


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
        fp = _compat(repr((CODE_VERSION, self.scenario, self.seed, self.days, self.strength, self.cons, self.world,
                           self.sensors, DEFAULT_CONFIG.preprocess, DEFAULT_CONFIG.features)))
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
    occ_true: Optional[np.ndarray] = None   # bool per operation window: someone is truly in the room (from the simulator)
    occ_label: Optional[np.ndarray] = None  # int8 per window: tracker belief 0 UNKNOWN, 1 EMPTY, 2 UNSURE, 3 OCCUPIED
    occ_conf: Optional[np.ndarray] = None   # occupancy confidence per window (nan for the original tracker)


def _path(cache_dir, spec: RunSpec) -> Optional[Path]:
    # ".v2": prepared runs now carry occupancy tracking; older entries without it are ignored, not reused
    return None if cache_dir is None else Path(cache_dir) / (spec.key() + ".v2.pkl")


_CLEAN_FIELDS = ("t", "temp", "hum", "dht_age", "pir", "pir_event", "pir_high_s", "door", "door_event", "ldr_raw", "ldr")
OCC_CODE = {"UNKNOWN": 0, "EMPTY": 1, "UNSURE": 2, "OCCUPIED": 3}


def _clean_path(cache_dir, spec: RunSpec) -> Optional[Path]:
    if cache_dir is None:
        return None
    fp = _compat(repr((CODE_VERSION, spec.scenario, spec.seed, spec.days, spec.strength, spec.world, spec.sensors, DEFAULT_CONFIG.preprocess)))
    return Path(cache_dir) / f"clean_{spec.scenario}_s{spec.seed}_{hashlib.sha1(fp.encode()).hexdigest()[:12]}.npz"


def clean_arrays(spec: RunSpec, cache_dir=None) -> dict:
    """
    The expensive part of a run: simulate and preprocess, stored as compact arrays (cached). It does not
    depend on the consensus or feature settings, so sweeping those never re-simulates.
    """
    path = _clean_path(cache_dir, spec)
    if path is not None and path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    src = ScenarioSource(spec.scenario, spec.seed, days=spec.days, world_cfg=spec.world,
                         sensors=spec.sensors, strength=spec.strength)
    pre = Preprocessor()
    cols = {k: [] for k in _CLEAN_FIELDS}
    nan = math.nan
    for r in strip_labels(src):
        c = pre.process(r)
        cols["t"].append(c.t); cols["temp"].append(nan if c.temp is None else c.temp)
        cols["hum"].append(nan if c.hum is None else c.hum); cols["dht_age"].append(nan if c.dht_age_s is None else c.dht_age_s)
        cols["pir"].append(c.pir); cols["pir_event"].append(int(c.pir_event)); cols["pir_high_s"].append(c.pir_high_s)
        cols["door"].append(c.door); cols["door_event"].append(c.door_event); cols["ldr_raw"].append(c.ldr_raw); cols["ldr"].append(c.ldr)
    dtypes = {"pir": np.int8, "pir_event": np.int8, "door": np.int8, "door_event": np.int8, "ldr_raw": np.int16}
    arrs = {k: np.array(v, dtype=dtypes.get(k, np.float64)) for k, v in cols.items()}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.replace(path)
    return arrs


def _iter_clean(arrs: dict):
    """Rebuild CleanReading objects from the cached arrays."""
    cols = [arrs[k].tolist() for k in _CLEAN_FIELDS]
    for t, tp, hm, ag, pir, pe, ph, door, de, lr, ldr in zip(*cols):
        yield CleanReading(t=t, temp=None if tp != tp else tp, hum=None if hm != hm else hm,
                           dht_age_s=None if ag != ag else ag, pir=pir, pir_event=bool(pe), pir_high_s=ph,
                           door=door, door_event=de, ldr_raw=lr, ldr=ldr)


def _presence_mask(spec: RunSpec, built) -> np.ndarray:
    """Ground truth: is someone in the room at each sample tick (from the simulator's schedule)."""
    world = World(spec.seed, spec.world, built.changes)
    dt = spec.world.sample_period_s
    n = int(round(built.duration_s / dt))
    mask = np.zeros(n, dtype=bool)
    for d in range(int(math.ceil((world.start_s + built.duration_s) / DAY_S))):
        _paint(mask, world.events(d).presence, world.start_s, dt)
    return mask


def prepare(spec: RunSpec, cache_dir=None) -> Prepared:
    """Analyse one run (cached). The first time is slow; sweeping consensus settings reuses the cleaned stream."""
    path = _path(cache_dir, spec)
    if path is not None and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    built = get_scenario(spec.scenario).build(spec.world, spec.days, spec.strength)
    arrs = clean_arrays(spec, cache_dir)
    eng = ConsensusEngine(spec.cons)
    cal, ts, xs, cons, truth = [], [], [], [], []
    calib_ended = False
    for w in extract_windows(_iter_clean(arrs), keep_samples=True):
        if not w.valid:
            continue
        if not calib_ended and w.t_end > built.calibration_s:
            eng.end_calibration()                      # the engine fixes what this room taught it (evidence model)
            calib_ended = True
        c = eng.evaluate(w)
        if w.t_end <= built.calibration_s:
            cal.append(w.x)
            continue
        atk = any(a.start_s < w.t_end and a.end_s > w.t_start for a in built.attacks)
        leg = any(ch.start_s < w.t_end and ch.label_end_s > w.t_start for ch in built.changes)
        ts.append(w.t_start); xs.append(w.x.copy()); cons.append(c)
        truth.append(TRUTH["attack"] if atk else (TRUTH["legit"] if leg else TRUTH["normal"]))
    t = np.array(ts)
    presence = _presence_mask(spec, built)
    dt, wlen = spec.world.sample_period_s, DEFAULT_CONFIG.features.window_s
    end_idx = np.clip(np.round((t + wlen) / dt).astype(int) - 1, 0, len(presence) - 1)
    prep = Prepared(spec, DEFAULT_CONFIG.features.feature_names, built.calibration_s, built.duration_s,
                    np.vstack(cal), t, np.vstack(xs), cons, np.array(truth, dtype=np.int8),
                    [(a.start_s, a.end_s) for a in built.attacks], [(ch.start_s, ch.label_end_s) for ch in built.changes],
                    occ_true=presence[end_idx],
                    occ_label=np.array([OCC_CODE[c.occupancy.value] for c in cons], dtype=np.int8),
                    occ_conf=np.array([c.occ_conf for c in cons], dtype=np.float64))
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
    gate: Optional[ConsensusConfig] = None      # None = the run's OWN consensus settings (the gate's rules depend on the evidence model)
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
    mu: Optional[np.ndarray] = None             # gated model's baseline mean after each window (n x d)
    mu_n: Optional[np.ndarray] = None           # naive twin's
    P_cal: Optional[np.ndarray] = None          # calibration precision matrix (to measure distances between baselines)

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
    gate = Gate(params.gate if params.gate is not None else prep.spec.cons, params.d_threshold)
    n = len(prep.t)
    D, Dn, lab = np.empty(n), np.empty(n), np.empty(n, dtype=np.int8)
    dr, drn = np.empty(n), np.empty(n)
    mu, mu_n = np.empty((n, model.d)), np.empty((n, model.d))
    index = {l: i for i, l in enumerate(LABELS)}
    for i in range(n):
        x = prep.X[i]
        D[i], Dn[i] = model.score(x), naive.score(x)
        dec = gate.decide(prep.cons[i], D[i])
        model.update(x, dec.alpha(ocfg))
        naive.update(x, naive_alpha)
        lab[i] = index[dec.label]
        dr[i], drn[i] = model.drift_distance(), naive.drift_distance()
        mu[i], mu_n[i] = model.mu, naive.mu
    return Replay(prep, params, D, Dn, lab, dr, drn, mu, mu_n, model._P_cal.copy())


def replay_cached(prep: Prepared, params: ReplayParams = ReplayParams(), cache_dir=None) -> Replay:
    """`replay`, cached on disk (keyed on the run and the parameters). Parameter sweeps repeat many replays."""
    if cache_dir is None:
        return replay(prep, params)
    key = hashlib.sha1(repr((CODE_VERSION, REPLAY_VERSION, prep.spec.key(), params)).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"replay_{key}.npz"
    if path.exists():
        z = np.load(path)
        return Replay(prep, params, z["D"], z["Dn"], z["label"], z["drift"], z["drift_n"], z["mu"], z["mu_n"], z["P_cal"])
    rep = replay(prep, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(tmp, D=rep.D, Dn=rep.Dn, label=rep.label, drift=rep.drift, drift_n=rep.drift_n,
                        mu=rep.mu, mu_n=rep.mu_n, P_cal=rep.P_cal)
    tmp.replace(path)
    return rep


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


def separation(attacked: Replay, benign: Replay) -> tuple[np.ndarray, np.ndarray]:
    """
    Distance, window by window, between the attacked run's baseline mean and the matched benign run's
    baseline mean (Mahalanobis, calibration covariance), for the gated and the naive model. This is the
    shift the attack itself caused. It is a more direct measure than 'excess drift' (the difference of the
    two runs' distances from calibration), because distances are not additive: a benign baseline that
    wanders in random directions makes that difference understate the attack.
    """
    if attacked.mu is None or benign.mu is None:
        raise ValueError("replays do not store baseline means; re-run them (replay cache version 2)")
    out = []
    for a, b in ((attacked.mu, benign.mu), (attacked.mu_n, benign.mu_n)):
        d = a - b
        out.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, attacked.P_cal, d), 0.0)))
    return out[0], out[1]


def excess(attacked: Replay, benign: Replay) -> dict:
    """Excess drift = attacked run minus its matched benign run (same seed, length and settings)."""
    if len(attacked.t) != len(benign.t) or not np.array_equal(attacked.t, benign.t):
        raise ValueError("runs are not matched: window timelines differ")
    last = _last_day(attacked.t)
    g = float(attacked.drift[last].mean() - benign.drift[last].mean())
    n = float(attacked.drift_n[last].mean() - benign.drift_n[last].mean())
    out = {"excess_gated": g, "excess_naive": n, "protection": n - g,
           "benign_drift": float(benign.drift[last].mean()), "benign_drift_naive": float(benign.drift_n[last].mean())}
    sg, sn = separation(attacked, benign)
    out.update({"sep_gated": float(sg[last].mean()), "sep_naive": float(sn[last].mean()),
                "sep_protection": float(sn[last].mean() - sg[last].mean())})
    return out


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
            benign_cache[s] = replay_cached(prepared[bs], params, cache_dir)
    for st in strengths:
        per_seed = []
        for s in seeds:
            spec = RunSpec(scenario, s, strength=st, **base)
            if prepared.get(spec) is None or s not in benign_cache:
                continue
            rep = replay_cached(prepared[spec], params, cache_dir)
            row = {"scenario": scenario, "seed": s, "strength": st, **metrics(rep), **excess(rep, benign_cache[s])}
            rows.append(row); per_seed.append(row)
        keys = ("excess_gated", "excess_naive", "protection", "sep_gated", "sep_naive", "sep_protection",
                "false_susp", "frozen_attack", "latency_min")
        summary[st] = {k: aggregate(r.get(k, math.nan) for r in per_seed) for k in keys}
        summary[st]["detected"] = (sum(1 for r in per_seed if r.get("detected")), len(per_seed))
        summary[st]["benign_drift"] = aggregate(r["benign_drift"] for r in per_seed)
    return {"rows": rows, "summary": summary, "missing": missing}


def format_excess(result: dict, scenario: str) -> str:
    mark = lambda a: str(a) + ("*" if a.significant else " ")
    rows = [(st, s) for st, s in result["summary"].items() if s["excess_gated"].n]
    lines = [f"{scenario}: poisoning effect versus a matched benign run (mean [95% CI] over seeds; '*' = interval excludes 0)", "",
             "(a) BASELINE SEPARATION: distance between the attacked baseline and the matched benign baseline",
             f"{'strength':>9} {'seeds':>6} | {'GATED':>24} {'NAIVE':>24} {'protection (naive-gated)':>26} | {'detected':>9} {'latency min':>12} {'attack frozen':>14} {'false SUSP':>11}"]
    for st, s in rows:
        det, lat = s["detected"], s["latency_min"]
        lines.append(f"{st:>9g} {s['excess_gated'].n:>6} | {mark(s['sep_gated']):>24} {mark(s['sep_naive']):>24} {mark(s['sep_protection']):>26} | "
                     f"{f'{det[0]}/{det[1]}':>9} {('%.0f' % lat.mean) if lat.n else 'never':>12} {s['frozen_attack'].mean:>14.0%} {s['false_susp'].mean:>11.2%}")
    lines += ["", "(b) EXCESS DRIFT: attacked run's drift from calibration minus the benign run's (difference of distances)",
              f"{'strength':>9} {'seeds':>6} | {'GATED':>24} {'NAIVE':>24} {'protection (naive-gated)':>26}"]
    for st, s in rows:
        lines.append(f"{st:>9g} {s['excess_gated'].n:>6} | {mark(s['excess_gated']):>24} {mark(s['excess_naive']):>24} {mark(s['protection']):>26}")
    b = next((s["benign_drift"] for _, s in rows), None)
    if b is not None:
        lines.append(f"(reference: the benign runs' own drift from calibration, last 24 h, gated model: {b.mean:.2f})")
    if result["missing"]:
        lines.append(f"[{result['missing']} run(s) not finished yet; run again to continue from the cache]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the learning-rate study: false alarms, adaptation speed and poisoning, per alpha
# ---------------------------------------------------------------------------
WINDOW_S = DEFAULT_CONFIG.features.window_s


def with_alpha(a_trusted: float, ratio: float = 0.2, base: ReplayParams = ReplayParams()) -> ReplayParams:
    """Replay settings with learning rate `a_trusted` (uncertain windows learn at ratio x that; the naive twin at a_trusted)."""
    return replace(base, online=replace(base.online, alpha_trusted=a_trusted, alpha_uncertain=a_trusted * ratio),
                   naive_alpha=a_trusted)


def block_rates(mask: np.ndarray, t: np.ndarray, t0: float, n_blocks: int, width: float = 86400.0,
                min_fill: float = 0.9) -> np.ndarray:
    """Fraction of `mask` that is true in consecutive blocks of `width` seconds from t0 (NaN if a block is under-filled)."""
    out = np.full(n_blocks, np.nan)
    for k in range(n_blocks):
        sel = (t >= t0 + k * width) & (t < t0 + (k + 1) * width)
        if sel.sum() >= min_fill * width / WINDOW_S:
            out[k] = mask[sel].mean()
    return out


def adaptation_block(excess_series: Sequence[float], tol: float = 0.01) -> Optional[int]:
    """First block from which the excess flagged rate stays <= tol; None if it never settles (censored)."""
    v = np.asarray(excess_series, dtype=float)
    ok = ~np.isnan(v)
    if not ok.any() or v[ok][-1] > tol:
        return None
    for k in range(len(v)):
        if np.all(v[k:][ok[k:]] <= tol):
            return k
    return None


def adaptation_experiment(kinds: Sequence[str], seeds: Sequence[int], alphas: Sequence[float], cache_dir,
                          days: float = 14.0, tol: float = 0.01, ratio: float = 0.2, workers: int = 1,
                          budget_s: Optional[float] = None, verbose: bool = True, strength: float = 1.0) -> dict:
    """
    How fast does the system stop flagging a LEGITIMATE environmental change, per learning rate?
    For each day after the change begins: flagged rate (D > threshold) in the changed run minus the
    same day of its matched benign run. 'Adapted' = the excess stays <= tol from some day on.
    """
    specs = [RunSpec(k, sd, days=days, strength=strength) for k in kinds for sd in seeds] + [RunSpec("normal", sd, days=days) for sd in seeds]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    missing = sum(1 for v in prepared.values() if v is None)
    rows = []
    for sd in seeds:
        bprep = prepared.get(RunSpec("normal", sd, days=days))
        if bprep is None:
            continue
        brep = {a: replay_cached(bprep, with_alpha(a, ratio), cache_dir) for a in alphas}
        for kind in kinds:
            prep = prepared.get(RunSpec(kind, sd, days=days, strength=strength))
            if prep is None:
                continue
            t0 = prep.legit_iv[0][0]
            nb = int((prep.duration_s - t0) // 86400.0)
            for a in alphas:
                rep, b = replay_cached(prep, with_alpha(a, ratio), cache_dir), brep[a]
                th = rep.params.d_threshold
                eg = block_rates(rep.D > th, rep.t, t0, nb) - block_rates(b.D > th, b.t, t0, nb)
                en = block_rates(rep.Dn > th, rep.t, t0, nb) - block_rates(b.Dn > th, b.t, t0, nb)
                during = prep.truth == TRUTH["legit"]
                rows.append(dict(kind=kind, seed=sd, alpha=a, n_blocks=nb, excess_gated=eg.tolist(), excess_naive=en.tolist(),
                                 adapt_gated=adaptation_block(eg, tol), adapt_naive=adaptation_block(en, tol),
                                 mean_excess_gated=float(np.nanmean(eg)), mean_excess_naive=float(np.nanmean(en)),
                                 peak_gated=float(np.nanmax(eg)),
                                 susp_during=float((rep.label[during] == 2).mean()) if during.any() else math.nan,
                                 unc_during=float((rep.label[during] == 1).mean()) if during.any() else math.nan))
    return {"rows": rows, "missing": missing, "tol": tol, "strength": strength}


def format_adaptation(res: dict) -> str:
    lines = [f"Adaptation to a legitimate change (strength {res.get('strength', 1):g}): excess flagged rate (changed run minus matched benign), "
             f"per day after the change; 'adapted' = stays <= {res['tol']:.0%} from some day on"]
    kinds = list(dict.fromkeys(r["kind"] for r in res["rows"]))
    alphas = sorted({r["alpha"] for r in res["rows"]}, reverse=True)
    for kind in kinds:
        lines.append(f"\n{kind}")
        lines.append(f"{'alpha':>8} {'tconst':>8} | {'GATED: mean excess [95% CI]':>30} {'days to adapt (per seed)':>26} | {'NAIVE: mean excess':>20} {'days (per seed)':>18} | {'SUSP during change':>19}")
        for a in alphas:
            rs = [r for r in res["rows"] if r["kind"] == kind and r["alpha"] == a]
            if not rs:
                continue
            show = lambda xs: ",".join("-" if x is None else str(x) for x in xs)
            lines.append(f"{a:>8g} {WINDOW_S / a / 3600:>6.1f} h | {str(aggregate(r['mean_excess_gated'] for r in rs)):>30} "
                         f"{show(r['adapt_gated'] for r in rs):>26} | {str(aggregate(r['mean_excess_naive'] for r in rs)):>20} "
                         f"{show(r['adapt_naive'] for r in rs):>18} | {np.nanmean([r['susp_during'] for r in rs]):>19.1%}")
    lines.append("\n('-' = never settled within the run; days counted from the start of the change)")
    if res["missing"]:
        lines.append(f"[{res['missing']} run(s) not finished yet]")
    return "\n".join(lines)


def longrun_experiment(seeds: Sequence[int], alphas: Sequence[float], cache_dir, days: float = 28.0, ratio: float = 0.2,
                       workers: int = 1, budget_s: Optional[float] = None, verbose: bool = True) -> dict:
    """False alarms and baseline drift over a long BENIGN run, by week, per learning rate."""
    specs = [RunSpec("normal", sd, days=days) for sd in seeds]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    rows = []
    for sp, prep in prepared.items():
        if prep is None:
            continue
        nb = int((prep.duration_s - prep.calib_s) // (7 * 86400.0))
        for a in alphas:
            rep = replay_cached(prep, with_alpha(a, ratio), cache_dir)
            th = rep.params.d_threshold
            wk = lambda m: block_rates(m, rep.t, prep.calib_s, nb, width=7 * 86400.0)
            drift_wk = np.array([rep.drift[(rep.t >= prep.calib_s + k * 7 * 86400.0) & (rep.t < prep.calib_s + (k + 1) * 7 * 86400.0)].mean() for k in range(nb)])
            rows.append(dict(seed=sp.seed, alpha=a, weeks=nb, flagged=float((rep.D > th).mean()), uncertain=float((rep.label == 1).mean()),
                             suspicious=float((rep.label == 2).mean()), flagged_wk=wk(rep.D > th).tolist(),
                             uncertain_wk=wk(rep.label == 1).tolist(), suspicious_wk=wk(rep.label == 2).tolist(), drift_wk=drift_wk.tolist()))
    return {"rows": rows, "missing": sum(1 for v in prepared.values() if v is None)}


def format_longrun(res: dict) -> str:
    lines = ["Long benign run: false alarms and baseline drift (mean over seeds; per-week values are weeks 1..N after calibration)"]
    lines.append(f"{'alpha':>8} {'tconst':>8} | {'D>th':>7} {'UNCERT':>7} {'SUSP':>7} | {'flagged % per week':>26} | {'drift (gated) per week':>26}")
    for a in sorted({r["alpha"] for r in res["rows"]}, reverse=True):
        rs = [r for r in res["rows"] if r["alpha"] == a]
        wk = lambda key, f: " ".join(f.format(x) for x in np.nanmean([r[key] for r in rs], axis=0))
        lines.append(f"{a:>8g} {WINDOW_S / a / 3600:>6.1f} h | {np.mean([r['flagged'] for r in rs]):>7.2%} {np.mean([r['uncertain'] for r in rs]):>7.2%} "
                     f"{np.mean([r['suspicious'] for r in rs]):>7.2%} | {wk('flagged_wk', '{:.1%}'):>26} | {wk('drift_wk', '{:.2f}'):>26}")
    lines.append(f"(seeds: {sorted({r['seed'] for r in res['rows']})})")
    if res["missing"]:
        lines.append(f"[{res['missing']} run(s) not finished yet]")
    return "\n".join(lines)


def poisoning_vs_alpha(scenario: str, seeds: Sequence[int], strength: float, alphas: Sequence[float], cache_dir,
                       days: Optional[float] = None, ratio: float = 0.2, workers: int = 1,
                       budget_s: Optional[float] = None) -> dict:
    """Excess drift (vs matched benign) of the gated and naive models for each learning rate."""
    out = {}
    for a in alphas:
        res = excess_experiment(scenario, seeds, [strength], cache_dir, with_alpha(a, ratio), days, workers, budget_s, verbose=False)
        out[a] = res["summary"][strength]
        out[a]["missing"] = res["missing"]
    return out


def format_poisoning_vs_alpha(res: dict, scenario: str, strength: float) -> str:
    mark = lambda a: str(a) + ("*" if a.significant else " ")
    lines = [f"{scenario} (strength {strength:g}): poisoning effect by learning rate (mean [95% CI] over seeds; '*' = interval excludes 0)",
             f"{'alpha':>8} {'tconst':>8} | {'SEPARATION gated':>22} {'SEPARATION naive':>22} {'protection':>20} | {'EXCESS DRIFT gated':>22} {'naive':>22} | {'detected':>9}"]
    for a in sorted(res, reverse=True):
        r = res[a]
        if r["excess_gated"].n == 0:
            continue
        d = r["detected"]
        lines.append(f"{a:>8g} {WINDOW_S / a / 3600:>6.1f} h | {mark(r['sep_gated']):>22} {mark(r['sep_naive']):>22} {mark(r['sep_protection']):>20} | "
                     f"{mark(r['excess_gated']):>22} {mark(r['excess_naive']):>22} | {f'{d[0]}/{d[1]}':>9}")
    return "\n".join(lines)


def post_attack_experiment(scenario: str, seeds: Sequence[int], alphas: Sequence[float], cache_dir, days: Optional[float] = None,
                           ratio: float = 0.2, workers: int = 1, budget_s: Optional[float] = None, verbose: bool = True,
                           window_h: float = 24.0) -> dict:
    """
    False SUSPICIOUS alarms on BENIGN windows in the `window_h` hours after an attack has ended, per
    learning rate. A fast baseline tracks the daily cycle, so freezing it during the attack leaves it
    stale afterwards. A FIXED window is used because the rate depends on how long you keep watching:
    the same run gives a much higher figure over a short tail than over a long one.
    """
    specs = [RunSpec(scenario, sd, days=days) for sd in seeds]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    out = {"rows": [], "missing": sum(1 for v in prepared.values() if v is None)}
    for sp, prep in prepared.items():
        if prep is None:
            continue
        end = max(e for _, e in prep.attack_iv)
        for a in alphas:
            rep = replay_cached(prep, with_alpha(a, ratio), cache_dir)
            post = (rep.t >= end) & (rep.t < end + window_h * 3600.0) & (prep.truth == TRUTH["normal"])
            pre = (rep.t < min(b for b, _ in prep.attack_iv)) & (prep.truth == TRUTH["normal"])
            out["rows"].append(dict(seed=sp.seed, alpha=a, post_false_susp=float((rep.label[post] == 2).mean()),
                                    pre_false_susp=float((rep.label[pre] == 2).mean()) if pre.any() else math.nan,
                                    n_post=int(post.sum()), detected=bool(((rep.label == 2) & (prep.truth == TRUTH["attack"])).any())))
    return out


def format_post_attack(res: dict, scenario: str) -> str:
    lines = [f"{scenario}: false SUSPICIOUS alarms on benign windows, before the attack and in the first 24 h after it ends (mean [95% CI] over seeds)"]
    lines.append(f"{'alpha':>8} {'tconst':>8} | {'before attack':>22} {'after attack':>26} | {'detected':>9}")
    for a in sorted({r["alpha"] for r in res["rows"]}, reverse=True):
        rs = [r for r in res["rows"] if r["alpha"] == a]
        lines.append(f"{a:>8g} {WINDOW_S / a / 3600:>6.1f} h | {str(aggregate(r['pre_false_susp'] for r in rs)):>22} "
                     f"{str(aggregate(r['post_false_susp'] for r in rs)):>26} | {sum(r['detected'] for r in rs)}/{len(rs)}")
    if res["missing"]:
        lines.append(f"[{res['missing']} run(s) not finished yet]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# occupancy tracking: how often is the room believed occupied when it is empty, and what does it cost?
# ---------------------------------------------------------------------------
def occupancy_metrics(prep: Prepared, after_s: float = -math.inf, min_spell_s: float = 3600.0) -> dict:
    """
    Compare the tracker's belief with the simulator's truth, window by window (from `after_s` on).
      false_occupied_empty  share of truly EMPTY windows believed OCCUPIED          (the self-reinforcement failure)
      false_unsure_empty    share of truly EMPTY windows believed UNSURE
      false_empty_occupied  share of truly OCCUPIED windows believed EMPTY          (the cost: false alarms follow)
      false_unsure_occupied share of truly OCCUPIED windows believed UNSURE
      recovery_*            minutes from someone leaving until the belief first becomes EMPTY, over absences of at
                            least `min_spell_s` (default 1 h); NaN if it never recovers before they return
    """
    if prep.occ_true is None:
        raise ValueError("run was prepared before occupancy tracking existed; re-prepare it")
    t, o, L = prep.t, prep.occ_true, prep.occ_label
    m = t >= after_s
    emp, occ = ~o & m, o & m
    frac = lambda sel, code: float((L[sel] == code).mean()) if sel.any() else math.nan
    dt = float(np.median(np.diff(t)))
    rec = []
    for i in np.flatnonzero(o[:-1] & ~o[1:]) + 1:                      # first window of each absence
        if t[i] < after_s:
            continue
        j = i + int(np.argmax(o[i:])) if o[i:].any() else len(o)        # first window of the next arrival
        if (j - i) * dt < min_spell_s:
            continue
        hit = np.flatnonzero(L[i:j] == 1)
        rec.append(hit[0] * dt / 60.0 if len(hit) else math.nan)
    rec = np.array(rec, dtype=float)
    known = rec[~np.isnan(rec)]
    return {"false_occupied_empty": frac(emp, 3), "false_unsure_empty": frac(emp, 2),
            "false_empty_occupied": frac(occ, 1), "false_unsure_occupied": frac(occ, 2),
            "recovery_median_min": float(np.median(known)) if len(known) else math.nan,
            "recovery_p90_min": float(np.percentile(known, 90)) if len(known) else math.nan,
            "recovery_never": float(np.isnan(rec).mean()) if len(rec) else math.nan, "n_absences": int(len(rec)),
            "n_empty": int(emp.sum())}


def horizon_configs(horizons_h: Sequence[float]) -> dict[str, ConsensusConfig]:
    """The original tracker plus the confidence tracker at each horizon (hours)."""
    from .config import OccupancyConfig
    out = {"original": ConsensusConfig()}
    for h in horizons_h:
        out[f"H={h:g}h"] = ConsensusConfig(occupancy=OccupancyConfig(pir_horizon_s=h * 3600.0))
    return out


def occupancy_experiment(configs: dict, scenario: str, seeds: Sequence[int], cache_dir, strength: float = 1.0,
                         days: Optional[float] = None, params: ReplayParams = ReplayParams(), workers: int = 1,
                         budget_s: Optional[float] = None, verbose: bool = True) -> dict:
    """
    For each tracker setting and seed: occupancy accuracy and false-alarm cost on the matched BENIGN run, and
    under the attack: occupancy accuracy once it has started, detection, windows frozen, and baseline separation.
    """
    specs = []
    for cfg in configs.values():
        for sd in seeds:
            atk = RunSpec(scenario, sd, days=days, strength=strength, cons=cfg)
            specs += [atk, atk.benign()]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    rows = []
    for label, cfg in configs.items():
        for sd in seeds:
            atk = RunSpec(scenario, sd, days=days, strength=strength, cons=cfg)
            pa, pb = prepared.get(atk), prepared.get(atk.benign())
            if pa is None or pb is None:
                continue
            ra, rb = replay_cached(pa, params, cache_dir), replay_cached(pb, params, cache_dir)
            start = min(a for a, _ in pa.attack_iv)
            ma, mb = metrics(ra), metrics(rb)
            oa, ob = occupancy_metrics(pa, after_s=start), occupancy_metrics(pb)
            rows.append({"config": label, "seed": sd, **{f"atk_{k}": v for k, v in oa.items()},
                         **{f"ben_{k}": v for k, v in ob.items()}, "ben_false_susp": mb["false_susp"],
                         "ben_uncertain": mb["frac_uncertain"], "detected": ma.get("detected", False),
                         "latency_min": ma.get("latency_min", math.nan), "frozen_attack": ma.get("frozen_attack", math.nan),
                         **excess(ra, rb)})
    return {"rows": rows, "configs": list(configs), "scenario": scenario,
            "missing": sum(1 for v in prepared.values() if v is None)}


def format_occupancy(res: dict) -> str:
    mark = lambda a: str(a) + ("*" if a.significant else " ")
    pct = lambda a: f"{a.mean:.1%}" if a.n else "n/a"
    lines = [f"{res['scenario']}: occupancy tracking and what it buys (mean over seeds; separation and protection also show the 95% CI)",
             f"{'tracker':>10} {'n':>2} | {'--- benign run ---':^40} | {'--- during the attack ---':^74}",
             f"{'':>10} {'':>2} | {'false OCC(empty)':>16} {'false EMPTY(occ)':>17} {'recover med/p90 min':>20} {'false SUSP':>11} | "
             f"{'false OCC(empty)':>16} {'detected':>9} {'latency h':>10} {'frozen':>7} {'separation (gated)':>21} {'protection':>22}"]
    for label in res["configs"]:
        rs = [r for r in res["rows"] if r["config"] == label]
        if not rs:
            continue
        g = lambda k: aggregate(r[k] for r in rs)
        rec_m, rec_p = g("ben_recovery_median_min"), g("ben_recovery_p90_min")
        lat = g("latency_min")
        lines.append(f"{label:>10} {len(rs):>2} | {pct(g('ben_false_occupied_empty')):>16} {pct(g('ben_false_empty_occupied')):>17} "
                     f"{(f'{rec_m.mean:.0f}/{rec_p.mean:.0f}' if rec_m.n else 'n/a'):>20} {pct(g('ben_false_susp')):>11} | "
                     f"{pct(g('atk_false_occupied_empty')):>16} {sum(bool(r['detected']) for r in rs)}/{len(rs):<7} "
                     f"{(f'{lat.mean / 60:.1f}' if lat.n else 'never'):>10} {pct(g('frozen_attack')):>7} "
                     f"{mark(g('sep_gated')):>21} {mark(g('sep_protection')):>22}")
    if res["missing"]:
        lines.append(f"[{res['missing']} run(s) not finished yet; run again to continue from the cache]")
    return "\n".join(lines)


def occupancy_longrun(configs: dict, seeds: Sequence[int], cache_dir, days: float = 21, params: ReplayParams = ReplayParams(),
                      workers: int = 1, budget_s: Optional[float] = None, verbose: bool = True) -> dict:
    """The false-alarm cost of each tracker setting on long BENIGN runs."""
    specs = [RunSpec("normal", sd, days=days, cons=cfg) for cfg in configs.values() for sd in seeds]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    rows = []
    for label, cfg in configs.items():
        for sd in seeds:
            prep = prepared.get(RunSpec("normal", sd, days=days, cons=cfg))
            if prep is None:
                continue
            rep = replay_cached(prep, params, cache_dir)
            m = metrics(rep)
            rows.append({"config": label, "seed": sd, "susp": m["frac_suspicious"], "uncertain": m["frac_uncertain"],
                         "n_susp": int((rep.label == 2).sum()), **occupancy_metrics(prep)})
    return {"rows": rows, "configs": list(configs), "days": days, "missing": sum(1 for v in prepared.values() if v is None)}


def format_occupancy_longrun(res: dict) -> str:
    lines = [f"Long benign runs ({res['days']:g} days): false-alarm cost of each tracker (mean over seeds)",
             f"{'tracker':>10} {'seeds':>5} | {'SUSPICIOUS':>11} {'(windows)':>10} {'UNCERTAIN':>10} | {'false EMPTY(occ)':>17} {'false UNSURE(occ)':>18} {'false OCC(empty)':>17} | {'recover med min':>16}"]
    for label in res["configs"]:
        rs = [r for r in res["rows"] if r["config"] == label]
        if not rs:
            continue
        mean = lambda k: float(np.nanmean([r[k] for r in rs]))
        lines.append(f"{label:>10} {len(rs):>5} | {mean('susp'):>11.2%} {mean('n_susp'):>10.0f} {mean('uncertain'):>10.2%} | {mean('false_empty_occupied'):>17.2%} "
                     f"{mean('false_unsure_occupied'):>18.2%} {mean('false_occupied_empty'):>17.2%} | {mean('recovery_median_min'):>16.0f}")
    if res["missing"]:
        lines.append(f"[{res['missing']} run(s) not finished yet]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# robustness across room conditions
# ---------------------------------------------------------------------------
def evidence_variants(occupancy=None) -> dict[str, ConsensusConfig]:
    """The current rules and the evidence model, switched on piece by piece. `occupancy` = tracker used by all."""
    from .config import EvidenceConfig
    base = dict(occupancy=occupancy)
    return {
        "current": ConsensusConfig(**base),
        "families": ConsensusConfig(evidence=EvidenceConfig(auto_light=False, temporal_bucket=False), **base),
        "+auto-light": ConsensusConfig(evidence=EvidenceConfig(temporal_bucket=False), **base),
        "+bucket (all)": ConsensusConfig(evidence=EvidenceConfig(), **base),
    }


def robustness_experiment(profiles: Sequence[str], variants: dict, attacks: Sequence[str], seeds: Sequence[int], cache_dir,
                          params: ReplayParams = ReplayParams(), days: Optional[float] = None, workers: int = 1,
                          budget_s: Optional[float] = None, verbose: bool = True) -> dict:
    """
    For every room profile and rule variant: false SUSPICIOUS / UNCERTAIN on benign data, and for each attack
    its detection latency and the share of attack windows frozen (plus baseline-separation protection).
    """
    from .scenarios import profile_configs
    spec = lambda scen, sd, cfg, prof: RunSpec(scen, sd, days=days, cons=cfg, world=profile_configs(prof)[0], sensors=profile_configs(prof)[1])
    specs = [spec(sc, sd, cfg, pr) for pr in profiles for cfg in variants.values() for sd in seeds for sc in ["normal", *attacks]]
    prepared = prepare_many(specs, cache_dir, workers, budget_s, verbose)
    rows = []
    for pr in profiles:
        for label, cfg in variants.items():
            for sd in seeds:
                bp = prepared.get(spec("normal", sd, cfg, pr))
                if bp is None:
                    continue
                rb = replay_cached(bp, params, cache_dir)
                mb = metrics(rb)
                row = {"profile": pr, "variant": label, "seed": sd, "ben_false_susp": mb["false_susp"],
                       "ben_uncertain": mb["frac_uncertain"], "ben_trusted": mb["frac_trusted"]}
                for a in attacks:
                    ap = prepared.get(spec(a, sd, cfg, pr))
                    if ap is None:
                        continue
                    ra = replay_cached(ap, params, cache_dir)
                    ma, ex = metrics(ra), excess(ra, rb)
                    row.update({f"{a}_detected": bool(ma.get("detected", False)), f"{a}_latency_min": ma.get("latency_min", math.nan),
                                f"{a}_frozen": ma.get("frozen_attack", math.nan), f"{a}_false_susp": ma["false_susp"],
                                f"{a}_sep": ex["sep_gated"], f"{a}_prot": ex["sep_protection"]})
                rows.append(row)
    return {"rows": rows, "profiles": list(profiles), "variants": list(variants), "attacks": list(attacks),
            "missing": sum(1 for v in prepared.values() if v is None)}


def format_robustness(res: dict) -> str:
    lines = ["Robustness across room conditions (mean over seeds). false SUSP / UNCERT = share of benign windows; per attack: "
             "detected seeds, latency, share of attack windows frozen"]
    for pr in res["profiles"]:
        lines.append(f"\n[{pr}]")
        head = f"{'rules':>14} {'n':>2} | {'false SUSP':>10} {'UNCERT':>7} |"
        for a in res["attacks"]:
            head += f" {a + ': found  latency  frozen':>36} |"
        lines.append(head)
        for v in res["variants"]:
            rs = [r for r in res["rows"] if r["profile"] == pr and r["variant"] == v]
            if not rs:
                continue
            mean = lambda k: float(np.nanmean([r[k] for r in rs if k in r])) if any(k in r for r in rs) else math.nan
            line = f"{v:>14} {len(rs):>2} | {mean('ben_false_susp'):>10.2%} {mean('ben_uncertain'):>7.2%} |"
            for a in res["attacks"]:
                got = [r for r in rs if f"{a}_detected" in r]
                lat = mean(f"{a}_latency_min")
                lat_s = "never" if math.isnan(lat) else (f"{lat:.0f} min" if lat < 600 else f"{lat / 60:.1f} h")
                line += f" {sum(r[f'{a}_detected'] for r in got)}/{len(got)}      {lat_s:>8}  {mean(f'{a}_frozen'):>6.0%}{'':>11} |" if got else f" {'n/a':>36} |"
            lines.append(line)
    if res["missing"]:
        lines.append(f"\n[{res['missing']} run(s) not finished yet; run again to continue from the cache]")
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
    lr = sub.add_parser("lr", help="learning-rate study: long benign false alarms and adaptation to legitimate changes")
    lr.add_argument("--seeds", default="0-2")
    lr.add_argument("--alphas", default="0.05,0.01,0.002,0.0005")
    lr.add_argument("--kinds", default="legit_warmer,legit_brighter,legit_busier")
    lr.add_argument("--benign-days", type=float, default=28.0)
    lr.add_argument("--legit-days", type=float, default=14.0)
    lr.add_argument("--strength", type=float, default=1.0, help="strength of the legitimate changes")
    lr.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    lr.add_argument("--cache", default="runs_cache")
    lr.add_argument("--budget", type=float, default=None)
    pa = sub.add_parser("post-attack", help="false alarms after an attack ends, per learning rate")
    pa.add_argument("--scenario", default="temp_spike")
    pa.add_argument("--seeds", default="0-2")
    pa.add_argument("--alphas", default="0.05,0.01,0.002,0.0005")
    pa.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    pa.add_argument("--cache", default="runs_cache")
    pa.add_argument("--budget", type=float, default=None)
    pz = sub.add_parser("poison-alpha", help="excess drift under poisoning, per learning rate")
    pz.add_argument("--scenario", default="gradual_poison")
    pz.add_argument("--seeds", default="0-4")
    pz.add_argument("--strength", type=float, default=1.0)
    pz.add_argument("--alphas", default="0.05,0.01,0.002,0.0005")
    pz.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    pz.add_argument("--cache", default="runs_cache")
    pz.add_argument("--budget", type=float, default=None)
    oc = sub.add_parser("occupancy", help="occupancy tracker comparison: original vs confidence tracker at several horizons")
    oc.add_argument("--scenario", default="gradual_poison")
    oc.add_argument("--seeds", default="0-2")
    oc.add_argument("--horizons", default="2,4,8", help="PIR-only horizons in hours")
    oc.add_argument("--long-seeds", default="0-1")
    oc.add_argument("--long-days", type=float, default=21.0)
    oc.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    oc.add_argument("--cache", default="runs_cache")
    oc.add_argument("--budget", type=float, default=None)
    rb = sub.add_parser("robustness", help="false alarms and detection across room profiles and rule variants")
    rb.add_argument("--profiles", default="default,dark_corridor,flickery_light,noisy_pir")
    rb.add_argument("--attacks", default="door_masked,gradual_poison")
    rb.add_argument("--seeds", default="0-1")
    rb.add_argument("--horizon", type=float, default=8.0, help="occupancy tracker horizon in hours (0 = original tracker)")
    rb.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    rb.add_argument("--cache", default="runs_cache")
    rb.add_argument("--budget", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "robustness":
        from .config import OccupancyConfig
        occ = OccupancyConfig(pir_horizon_s=a.horizon * 3600.0) if a.horizon > 0 else None
        print(format_robustness(robustness_experiment(a.profiles.split(","), evidence_variants(occ), a.attacks.split(","),
                                                      _parse_seeds(a.seeds), a.cache, workers=a.workers, budget_s=a.budget)))
        return 0
    if a.cmd == "occupancy":
        cfgs = horizon_configs([float(x) for x in a.horizons.split(",")])
        print(format_occupancy(occupancy_experiment(cfgs, a.scenario, _parse_seeds(a.seeds), a.cache, workers=a.workers, budget_s=a.budget)))
        print()
        print(format_occupancy_longrun(occupancy_longrun(cfgs, _parse_seeds(a.long_seeds), a.cache, a.long_days, workers=a.workers, budget_s=a.budget)))
        return 0
    if a.cmd == "post-attack":
        res = post_attack_experiment(a.scenario, _parse_seeds(a.seeds), [float(x) for x in a.alphas.split(",")], a.cache,
                                     workers=a.workers, budget_s=a.budget)
        print(format_post_attack(res, a.scenario))
        return 0
    if a.cmd == "poison-alpha":
        res = poisoning_vs_alpha(a.scenario, _parse_seeds(a.seeds), a.strength, [float(x) for x in a.alphas.split(",")], a.cache,
                                 workers=a.workers, budget_s=a.budget)
        print(format_poisoning_vs_alpha(res, a.scenario, a.strength))
        return 0
    if a.cmd == "lr":
        seeds, alphas = _parse_seeds(a.seeds), [float(x) for x in a.alphas.split(",")]
        print(format_longrun(longrun_experiment(seeds, alphas, a.cache, a.benign_days, workers=a.workers, budget_s=a.budget)))
        print()
        print(format_adaptation(adaptation_experiment(a.kinds.split(","), seeds, alphas, a.cache, a.legit_days,
                                                      workers=a.workers, budget_s=a.budget, strength=a.strength)))
        return 0
    res = excess_experiment(a.scenario, _parse_seeds(a.seeds), [float(x) for x in a.strengths.split(",")], a.cache,
                            days=a.days, workers=a.workers, budget_s=a.budget)
    print(format_excess(res, a.scenario))
    return 0


if __name__ == "__main__":
    sys.exit(_main())