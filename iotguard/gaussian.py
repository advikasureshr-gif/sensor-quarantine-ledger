"""
gaussian.py - the learned behavioural baseline: an online multivariate Gaussian (mu, Sigma)
and the Mahalanobis anomaly score. This is the "Update mu/Sigma" and "Mahalanobis score"
part of the flow.

It knows nothing about trust, consensus or attacks. It exposes exactly two things the gate
needs: `score(x)` ("how unusual is this window?") and `update(x, alpha)` ("learn from this
window at this rate"; alpha = 0 freezes the model). The decision about WHICH alpha to use
belongs to the gate; running the same class with alpha always > 0 is the naive baseline.

Model details
-------------
* Features are first transformed: `log1p_features` (skewed counts/spreads) go through log1p;
  `exclude_features` are dropped. mu and Sigma live in that transformed space.
* Effective covariance used for scoring:
      Sigma_eff = (1 - shrinkage) * Sigma + shrinkage * diag(Sigma) + diag(min_std**2)
  The shrinkage stabilises the inverse; the floor `min_std` stops a nearly-constant feature
  from turning every ordinary deviation into an enormous score.
* Online update (exponentially weighted):
      d = x - mu;   mu += alpha * d;   Sigma = (1 - alpha) * (Sigma + alpha * d d^T)
* Drift cap: after each update mu is clipped to within `drift_cap_std` calibration standard
  deviations of the calibrated mean (per feature).

Research metric: `drift_distance()` is how far the current baseline has moved from the
calibrated one (Mahalanobis distance between mu and mu_cal under the calibration covariance).
Plotted over time for the naive and the gated model, this is the baseline-drift figure.

Only finite, valid windows may be passed in; invalid windows must be skipped by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, OnlineStatsConfig


@dataclass(frozen=True, eq=False)
class BaselineSnapshot:
    """An immutable copy of the model state (for versioning, comparison and the ledger)."""
    names: tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray
    mu_cal: np.ndarray
    sigma_cal: np.ndarray
    n_updates: int


class GaussianModel:
    def __init__(self, cfg: OnlineStatsConfig = DEFAULT_CONFIG.online_stats,
                 feature_names: Sequence[str] = DEFAULT_CONFIG.features.feature_names) -> None:
        feature_names = tuple(feature_names)
        for label, names in (("log1p_features", cfg.log1p_features), ("exclude_features", cfg.exclude_features)):
            unknown = [n for n in names if n not in feature_names]
            if unknown:
                raise ValueError(f"{label} contains unknown features: {unknown}")
        self.cfg = cfg
        self.all_names = feature_names
        self.names = tuple(n for n in feature_names if n not in cfg.exclude_features)
        if not self.names:
            raise ValueError("all features are excluded")
        missing = [n for n in self.names if n not in cfg.min_std]
        if missing or any(cfg.min_std[n] <= 0 for n in self.names):
            raise ValueError(f"min_std must be defined and > 0 for every modelled feature (missing: {missing})")
        if not 0.0 <= cfg.shrinkage < 1.0:
            raise ValueError("shrinkage must be in [0, 1)")
        if cfg.drift_cap_std is not None and cfg.drift_cap_std <= 0:
            raise ValueError("drift_cap_std must be > 0 or None")

        self._cols = [feature_names.index(n) for n in self.names]
        self._log = np.array([n in cfg.log1p_features for n in self.names])
        self._floor2 = np.array([cfg.min_std[n] ** 2 for n in self.names])
        self.d = len(self.names)

        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.mu_cal: Optional[np.ndarray] = None
        self.sigma_cal: Optional[np.ndarray] = None
        self._std_cal: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None          # inverse of Sigma_eff, cached
        self._P_sub: dict[tuple, np.ndarray] = {}     # inverses of Sigma_eff restricted to a subset of features
        self._P_cal: Optional[np.ndarray] = None
        self.n_updates = 0

    # ---- helpers -------------------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self.mu is not None

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() with calibration windows first")

    def _eff(self, sigma: np.ndarray) -> np.ndarray:
        lam = self.cfg.shrinkage
        return (1.0 - lam) * sigma + lam * np.diag(np.diag(sigma)) + np.diag(self._floor2)

    def _transform(self, x) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.shape[-1] != len(self.all_names):
            raise ValueError(f"expected {len(self.all_names)} features, got shape {a.shape}")
        z = a[..., self._cols].copy()
        if not np.isfinite(z).all():
            raise ValueError("feature vector contains NaN/inf; skip invalid windows before modelling")
        z[..., self._log] = np.log1p(np.maximum(z[..., self._log], 0.0))
        return z

    def _refresh(self) -> None:
        self._P = np.linalg.inv(self._eff(self.sigma))
        self._P_sub = {}

    # ---- calibration -----------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> None:
        """Build the initial baseline from calibration windows, shape (n_windows, n_all_features)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (windows x features)")
        if X.shape[0] < 5 * self.d:
            raise ValueError(f"need at least {5 * self.d} windows to fit {self.d} features, got {X.shape[0]}")
        Z = self._transform(X)
        self.mu = Z.mean(axis=0)
        self.sigma = np.cov(Z, rowvar=False).reshape(self.d, self.d)
        self.mu_cal, self.sigma_cal = self.mu.copy(), self.sigma.copy()
        eff_cal = self._eff(self.sigma_cal)
        self._std_cal = np.sqrt(np.diag(eff_cal))
        self._P_cal = np.linalg.inv(eff_cal)
        self.n_updates = 0
        self._refresh()

    # ---- scoring ---------------------------------------------------------------------------
    def score(self, x, exclude=()) -> float:
        """
        Mahalanobis distance of one feature vector from the current baseline. `exclude` names features to leave
        out (a quarantined sensor's): the score is then the distance in the remaining features' own marginal
        distribution, so a faulty sensor cannot make healthy ones look anomalous.
        """
        self._require_fit()
        keep = self._keep(exclude)
        dv = self._transform(x) - self.mu
        if keep is not None:
            dv = dv[list(keep)]
        return float(np.sqrt(max(dv @ self._precision(keep) @ dv, 0.0)))

    def _keep(self, exclude) -> Optional[tuple[int, ...]]:
        """Indices of the modelled features that are NOT excluded; None if nothing is excluded."""
        if not exclude:
            return None
        unknown = [n for n in exclude if n not in self.all_names]
        if unknown:
            raise ValueError(f"unknown features to exclude: {unknown}")
        ex = set(exclude)
        keep = tuple(i for i, n in enumerate(self.names) if n not in ex)
        if not keep:
            raise ValueError("cannot exclude every modelled feature")
        return None if len(keep) == self.d else keep

    def _precision(self, keep: Optional[tuple[int, ...]]) -> np.ndarray:
        if keep is None:
            return self._P
        if keep not in self._P_sub:
            idx = np.array(keep)
            self._P_sub[keep] = np.linalg.inv(self._eff(self.sigma)[np.ix_(idx, idx)])
        return self._P_sub[keep]

    def score_many(self, X) -> np.ndarray:
        self._require_fit()
        dv = self._transform(X) - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dv, self._P, dv), 0.0))

    def contributions(self, x, exclude=()) -> dict[str, float]:
        """Each (non-excluded) feature's share of D**2 (they sum to D**2). Can be negative for correlated features."""
        self._require_fit()
        keep = self._keep(exclude)
        dv = self._transform(x) - self.mu
        if keep is None:
            return dict(zip(self.names, (dv * (self._P @ dv)).tolist()))
        dk = dv[list(keep)]
        return dict(zip([self.names[i] for i in keep], (dk * (self._precision(keep) @ dk)).tolist()))

    # ---- learning ----------------------------------------------------------------------------
    def update(self, x, alpha: float, exclude=()) -> bool:
        """
        Learn from one window at rate `alpha` (0 = frozen: nothing changes, bit for bit).
        `exclude` names features of a quarantined sensor: they are replaced by the baseline's own mean, so the
        sensor cannot move the baseline (its variance shrinks very slowly, as for any window that agrees with the mean).
        Returns True if the drift cap clipped the mean.
        """
        self._require_fit()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        z = self._transform(x)               # validate even when frozen
        keep = self._keep(exclude)
        if keep is not None and alpha > 0.0:
            drop = [i for i in range(self.d) if i not in set(keep)]
            z = z.copy()
            z[drop] = self.mu[drop]
        if alpha == 0.0:
            return False
        dv = z - self.mu
        self.mu = self.mu + alpha * dv
        sig = (1.0 - alpha) * (self.sigma + alpha * np.outer(dv, dv))
        self.sigma = 0.5 * (sig + sig.T)
        capped = False
        cap = self.cfg.drift_cap_std
        if cap is not None:
            lo, hi = self.mu_cal - cap * self._std_cal, self.mu_cal + cap * self._std_cal
            clipped = np.clip(self.mu, lo, hi)
            capped = bool((clipped != self.mu).any())
            self.mu = clipped
        self.n_updates += 1
        self._refresh()
        return capped

    # ---- drift metrics ---------------------------------------------------------------------------
    def drift(self) -> dict[str, float]:
        """Per-feature baseline movement since calibration, in calibration standard deviations."""
        self._require_fit()
        return dict(zip(self.names, ((self.mu - self.mu_cal) / self._std_cal).tolist()))

    def drift_distance(self) -> float:
        """Mahalanobis distance between the current mean and the calibrated mean (calibration covariance)."""
        self._require_fit()
        dv = self.mu - self.mu_cal
        return float(np.sqrt(max(dv @ self._P_cal @ dv, 0.0)))

    # ---- state ---------------------------------------------------------------------------------------
    def snapshot(self) -> BaselineSnapshot:
        self._require_fit()
        return BaselineSnapshot(self.names, self.mu.copy(), self.sigma.copy(), self.mu_cal.copy(),
                                self.sigma_cal.copy(), self.n_updates)

    def restore(self, snap: BaselineSnapshot) -> None:
        if snap.names != self.names:
            raise ValueError("snapshot was taken with a different feature set")
        self.mu, self.sigma = snap.mu.copy(), snap.sigma.copy()
        self.mu_cal, self.sigma_cal = snap.mu_cal.copy(), snap.sigma_cal.copy()
        eff_cal = self._eff(self.sigma_cal)
        self._std_cal = np.sqrt(np.diag(eff_cal))
        self._P_cal = np.linalg.inv(eff_cal)
        self.n_updates = snap.n_updates
        self._refresh()

    def copy(self) -> "GaussianModel":
        """An independent model with the same state (e.g. to run a naive twin next to a gated model)."""
        m = GaussianModel(self.cfg, self.all_names)
        if self.fitted:
            m.restore(self.snapshot())
        return m