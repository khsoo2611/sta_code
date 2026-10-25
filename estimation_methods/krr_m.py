"""Matern kernel ridge regression.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
folds : list
    Training and validation indices for cross-validation (CV).
sample : Sample or None
    Direct methods pass sample to use beta-based CV.
    EASE passes None to use regression prediction MSE.

Methods and Functions
---------------------
fit selects the length scale and ridge value by CV.
predict gives a regression prediction for each new observation.
matern computes the kernel values.
matern_ell finds the length scale for a given kernel value.
"""

import numpy as np
from scipy.special import gamma, kv

from utils import add_intercept, krr_cv_scores, krr_fit, median_sq_dist, sq_dist

MATERN_Q = np.array([0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.170, 0.280,
                     0.450, 0.650, 0.800, 0.900, 0.960, 0.985, 0.995])
MATERN_GAMMA_FRACTION = np.linspace(0.10, 0.99, 10)  # gamma = fraction * 2t/d


def matern(D, ell, nu):
    z = np.sqrt(2.0 * nu) * D / ell
    K = np.ones_like(z)
    far = z > np.sqrt(np.finfo(float).eps)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        v = z[far] ** nu * kv(nu, z[far]) / (gamma(nu) * 2.0 ** (nu - 1.0))
    K[far] = np.where(np.isfinite(v), v, 0.0)
    return np.clip(K, 0.0, 1.0)


def matern_ell(distance, nu, q):
    """Length scale at which the kernel equals q at `distance` (bisection)."""
    # start bracketing a little below sqrt(1/6), the RMS distance of two U(0, 1) points
    lo, hi = 1e-12, max(distance, np.sqrt(1 / 6) / 1.30, 1e-6)
    while matern(np.array([distance]), hi, nu)[0] < q:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if matern(np.array([distance]), mid, nu)[0] < q:
            lo = mid
        else:
            hi = mid
    return hi


class KRR_M:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self, folds, sample=None):
        """Select the length scale and ridge value using the supplied CV folds."""
        X, y = self.X, self.y
        n, d = X.shape
        t = d // 2 + 1
        nu = t - d / 2
        d_med = np.sqrt(median_sq_dist(X))
        ells = np.array([matern_ell(d_med, nu, q) for q in MATERN_Q])
        ridges = n * n ** -(MATERN_GAMMA_FRACTION * (2 * t / d))
        D = np.sqrt(sq_dist(X, X))
        D_u = None if sample is None else np.sqrt(sq_dist(sample.X_u, X))
        X1 = add_intercept(X)
        same_ridges = lambda eigvals, n_train: ridges
        scores = []
        for ell in ells:
            K_u = None if sample is None else matern(D_u, ell, nu)
            scores.append(krr_cv_scores(matern(D, ell, nu), y, folds, same_ridges, K_u, X1, sample))
        scores = np.array(scores)
        i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
        self.tuning = {"q": MATERN_Q[i], "ell": ells[i], "ridge": ridges[j]}
        self.cv_score = scores[i, j]
        self.nu = nu
        K = matern(np.sqrt(sq_dist(X, X)), self.tuning["ell"], self.nu)
        self.alpha = krr_fit(K, y, self.tuning["ridge"])
        return self

    def predict(self, X_new):
        K_new = matern(np.sqrt(sq_dist(X_new, self.X)), self.tuning["ell"], self.nu)
        return K_new @ self.alpha
