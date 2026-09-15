"""KRR-M and KRR-R: kernel ridge regression imputation with Matern and RBF kernels.

The kernel scale is tuned through q, the kernel value at the median pairwise
distance of the training covariates. The ridge grid differs by kernel:
Matern uses lambda = n^(-gamma) on a grid of gamma, RBF targets a fraction of
the effective degrees of freedom.
"""

import numpy as np
from scipy import linalg
from scipy.special import gamma, kv

from .utils import add_intercept, solve

MATERN_Q = np.array([0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.170, 0.280,
                     0.450, 0.650, 0.800, 0.900, 0.960, 0.985, 0.995])
MATERN_GAMMA_FRACTION = np.linspace(0.10, 0.99, 10)  # gamma = fraction * 2t/d
RBF_Q = np.geomspace(0.02, 0.85, 10)
RBF_DF_FRACTION = np.geomspace(0.025, 0.75, 10)


def sq_dist(A, B):
    D2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def median_sq_dist(X):
    D2 = sq_dist(X, X)[np.triu_indices(len(X), k=1)]
    D2 = D2[np.isfinite(D2) & (D2 > 0.0)]
    return float(np.median(D2)) if D2.size else 1.0


# Matern kernel

def matern_nu(d):
    # Sobolev order t = floor(d/2) + 1, so nu = t - d/2 is 1 for even d and 1/2 for odd d
    return d // 2 + 1 - d / 2


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


def matern_ridges(n, d):
    t = d // 2 + 1
    return n * n ** -(MATERN_GAMMA_FRACTION * (2 * t / d))


# RBF kernel: ridge from effective degrees of freedom

def ridge_for_df(eigvals, target_df):
    """Ridge values with sum(l / (l + ridge)) = target_df, found by bisection."""
    lam = np.maximum(eigvals[np.isfinite(eigvals)], 0.0)
    target = np.asarray(target_df, dtype=float)
    ridge = np.full_like(target, 1e-12)  # used when the target is not below rank(K)
    active = target < np.sum(lam > 0.0)
    goal = target[active]
    df = lambda r: np.sum(lam[:, None] / (lam[:, None] + r[None, :]), axis=0)
    lo = np.zeros_like(goal)
    hi = np.full_like(goal, max(float(np.max(lam)), 1.0))
    while np.any(df(hi) > goal):
        hi[df(hi) > goal] *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        above = df(mid) > goal
        lo[above] = mid[above]
        hi[~above] = mid[~above]
    ridge[active] = np.maximum(hi, 1e-12)
    return ridge


# Cross-validation and fitting

def eigh_psd(K):
    lam, V = np.linalg.eigh(0.5 * (K + K.T))
    return np.maximum(lam, 0.0), V


def cv_errors(K, y, folds, ridge_grid, K_u=None, X1=None, sample=None):
    """Mean validation MSE over the folds, one value per ridge.

    ridge_grid(eigvals, n_train) gives the ridges to try in a fold. When K_u,
    the kernel between U and the training points, is given, the fit on U is
    projected onto (1, X_u) and X1 @ beta_hat is scored, as for the direct
    estimators; otherwise the kernel fit itself is scored.
    """
    total = 0.0
    for train, val in folds:
        lam, V = eigh_psd(K[np.ix_(train, train)])
        ridges = np.maximum(ridge_grid(lam, train.size), 1e-12)
        alpha = V @ ((V.T @ y[train])[:, None] / (lam[:, None] + ridges[None, :]))
        if K_u is None:
            pred = K[np.ix_(val, train)] @ alpha
        else:
            beta = solve(sample.Gamma_u, sample.X_u1.T @ K_u[:, train] / len(K_u) @ alpha)
            pred = X1[val] @ beta
        mse = np.mean((pred - y[val, None]) ** 2, axis=0)
        total = total + np.where(np.isfinite(mse), mse, np.inf)
    return total / len(folds)


def select_matern(X, y, folds, sample=None):
    n, d = X.shape
    nu = matern_nu(d)
    d_med = np.sqrt(median_sq_dist(X))
    ells = np.array([matern_ell(d_med, nu, q) for q in MATERN_Q])
    ridges = matern_ridges(n, d)
    D = np.sqrt(sq_dist(X, X))
    D_u = None if sample is None else np.sqrt(sq_dist(sample.X_u, X))
    X1 = add_intercept(X)
    same_ridges = lambda eigvals, n_train: ridges
    scores = []
    for ell in ells:
        K_u = None if sample is None else matern(D_u, ell, nu)
        scores.append(cv_errors(matern(D, ell, nu), y, folds, same_ridges, K_u, X1, sample))
    scores = np.array(scores)
    i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
    return {"q": MATERN_Q[i], "ell": ells[i], "ridge": ridges[j]}, scores[i, j]


def select_rbf(X, y, folds, sample=None):
    n = len(X)
    D2 = sq_dist(X, X)
    D2_u = None if sample is None else sq_dist(sample.X_u, X)
    X1 = add_intercept(X)
    gammas = -np.log(RBF_Q) / median_sq_dist(X)
    df_ridges = lambda eigvals, n_train: ridge_for_df(eigvals, RBF_DF_FRACTION * n_train)
    scores = []
    for g in gammas:
        K_u = None if sample is None else np.exp(-g * D2_u)
        scores.append(cv_errors(np.exp(-g * D2), y, folds, df_ridges, K_u, X1, sample))
    scores = np.array(scores)
    i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
    lam, _ = eigh_psd(np.exp(-gammas[i] * D2))
    ridge = ridge_for_df(lam, [RBF_DF_FRACTION[j] * n])[0]
    tuning = {"q": RBF_Q[i], "gamma": gammas[i], "df_fraction": RBF_DF_FRACTION[j], "ridge": ridge}
    return tuning, scores[i, j]


def krr_fit(K, y, ridge):
    A = K + ridge * np.eye(len(K))
    try:
        return linalg.solve(A, y, assume_a="pos", check_finite=False)
    except np.linalg.LinAlgError:
        return solve(A, y)


def impute(X, y, X_new, folds, kind, sample=None):
    """Tune by CV, fit on (X, y) and predict at X_new. kind is "matern" or "rbf".

    With `sample`, tuning uses the beta-based criterion of the direct
    estimators; otherwise (EASE nuisance fits) the plain prediction error.
    """
    if kind == "matern":
        tuning, score = select_matern(X, y, folds, sample)
        nu = matern_nu(X.shape[1])
        K = matern(np.sqrt(sq_dist(X, X)), tuning["ell"], nu)
        alpha = krr_fit(K, y, tuning["ridge"])
        K_new = matern(np.sqrt(sq_dist(X_new, X)), tuning["ell"], nu)
    else:
        tuning, score = select_rbf(X, y, folds, sample)
        K = np.exp(-tuning["gamma"] * sq_dist(X, X))
        alpha = krr_fit(K, y, tuning["ridge"])
        K_new = np.exp(-tuning["gamma"] * sq_dist(X_new, X))
    return K_new @ alpha, tuning, score


def run(sample, kind):
    m_u, tuning, score = impute(sample.X, sample.y, sample.X_u, sample.folds, kind, sample)
    return {"m_u": m_u, "tuning": tuning, "cv_score": score}
