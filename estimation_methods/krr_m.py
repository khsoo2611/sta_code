"""Matern kernel ridge regression.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
sample : Sample
    Data and CV folds used to score the projected coefficients.

Methods and Functions
---------------------
fit selects the length scale and ridge value by CV.
predict gives a regression prediction for each new observation.
matern computes the kernel values directly.
matern_cv uses interpolation during CV when nu = 1.
matern_ell finds the length scale for a given kernel value.
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import k1

from utils import add_intercept, krr_cv_scores, krr_fit, median_sq_dist, sq_dist

MATERN_Q = np.array([0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.170, 0.280,
                     0.450, 0.650, 0.800, 0.900, 0.960, 0.985, 0.995])
MATERN_GAMMA_FRACTION = np.linspace(0.10, 0.99, 10)  # gamma = fraction * 2t/d


# Use a fine log grid for CV. Use direct values above Z_MAX.
LOG_STEP = 1.0 / 2048.0
Z_MIN = np.sqrt(np.finfo(float).eps)
Z_MAX = 64.0


def matern(D, ell, nu):
    z = np.sqrt(2.0 * nu) * D / ell
    K = np.ones_like(z)
    far = z > Z_MIN
    z_far = z[far]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        # nu = t - d/2 is 1/2 for odd d and 1 for even d (EASE: r = 2, so nu = 1).
        if nu == 0.5:
            v = np.exp(-z_far)
        elif nu == 1.0:
            v = z_far * k1(z_far)
        else:
            raise ValueError("nu must be 0.5 or 1")
    K[far] = np.where(np.isfinite(v), v, 0.0)
    return np.clip(K, 0.0, 1.0)


@lru_cache(maxsize=1)
def _lookup():
    # Add 32 points at each end to reduce spline edge effects.
    start = np.floor(np.log(Z_MIN) / LOG_STEP) * LOG_STEP - 32 * LOG_STEP
    stop = np.ceil(np.log(Z_MAX) / LOG_STEP) * LOG_STEP + 32 * LOG_STEP
    size = int(round((stop - start) / LOG_STEP)) + 1
    z = np.exp(start + LOG_STEP * np.arange(size))
    coefficients = spline_filter(z * k1(z), order=3, mode="mirror")
    coefficients.flags.writeable = False
    return start, coefficients


def matern_cv(D, ell, nu):
    """Kernel for CV. Use a spline for nu = 1 and direct values above Z_MAX."""
    if nu != 1.0:
        return matern(D, ell, nu)
    z = np.sqrt(2.0 * nu) * D / ell
    K = np.ones_like(z)
    far = z > Z_MIN
    z_far = z[far]
    values = np.empty_like(z_far)
    inside = np.isfinite(z_far) & (z_far <= Z_MAX)
    if np.any(inside):
        start, coefficients = _lookup()
        coordinates = ((np.log(z_far[inside]) - start) / LOG_STEP)[None, :]
        values[inside] = map_coordinates(
            coefficients, coordinates, order=3, mode="mirror", prefilter=False,
        )
    if np.any(~inside):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values[~inside] = z_far[~inside] * k1(z_far[~inside])
    K[far] = np.where(np.isfinite(values), values, 0.0)
    return np.clip(K, 0.0, 1.0)


def matern_ell(distance, nu, q):
    """Length scale at which the kernel equals q at `distance` (bisection)."""
    # Start with a positive upper bound and expand it if needed.
    lo, hi = 1e-12, max(distance, np.sqrt(1 / 6) / 1.30)
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

    def fit(self, sample):
        """Select the length scale and ridge value using sample.folds."""
        X, y = self.X, self.y
        n, d = X.shape
        t = d // 2 + 1
        nu = t - d / 2
        d_med = np.sqrt(median_sq_dist(X))
        ells = np.array([matern_ell(d_med, nu, q) for q in MATERN_Q])
        ridges = n * n ** -(MATERN_GAMMA_FRACTION * (2 * t / d))
        D = np.sqrt(sq_dist(X, X))
        D_u = np.sqrt(sq_dist(sample.X_u, X))
        X1 = add_intercept(X)
        scores = []
        for ell in ells:
            # Use interpolation only during CV.
            # Use the direct kernel for the final fit and prediction.
            K_u = matern_cv(D_u, ell, nu)
            scores.append(krr_cv_scores(matern_cv(D, ell, nu), ridges, K_u, X1, sample))
        scores = np.array(scores)
        i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
        self.tuning = {"q": MATERN_Q[i], "ell": ells[i], "ridge": ridges[j]}
        self.cv_score = scores[i, j]
        self.nu = nu
        K = matern(D, self.tuning["ell"], self.nu)
        self.alpha = krr_fit(K, y, self.tuning["ridge"])
        return self

    def predict(self, X_new):
        K_new = matern(np.sqrt(sq_dist(X_new, self.X)), self.tuning["ell"], self.nu)
        return K_new @ self.alpha
