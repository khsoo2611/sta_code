"""ATA-SBF: smooth backfitting of an additive local-constant model.

Covariates lie in [0, 1]. Each component m_j is stored on an equally spaced
grid and linearly interpolated between grid points.
"""

import numpy as np

from .utils import add_intercept, solve


def bandwidth_grid(n):
    """20 equally spaced bandwidths strictly inside (n^(-1/2), n^(-1/4))."""
    lo, hi = n ** -0.5, n ** -0.25
    return lo + np.arange(1, 21) / 21 * (hi - lo)


def trapezoid_weights(t):
    dt = np.diff(t)
    return (np.append(dt, 0.0) + np.append(0.0, dt)) / 2.0


def boundary_kernel(X, h, ngrid):
    """Grid t, its quadrature weights, and K[j, g, i] = K_h(t_g - X_ij) divided by
    the integral of K_h(u - X_ij) over u in [0, 1] (Epanechnikov kernel)."""
    n, d = X.shape
    t = np.linspace(0.0, 1.0, ngrid)
    K = np.empty((d, ngrid, n))
    for j in range(d):
        x = X[:, j]
        u = (t[:, None] - x[None, :]) / h
        K[j] = 0.75 * (1.0 - u**2) * (np.abs(u) < 1.0) / h
        a = (np.maximum(0.0, x - h) - x) / h
        b = (np.minimum(1.0, x + h) - x) / h
        K[j] /= 0.75 * ((b - a) - (b**3 - a**3) / 3.0)
    return t, trapezoid_weights(t), K


def backfit(K, grid_w, density, wy, K_proj=None, max_iter=100, tol=1e-8):
    """Smooth backfitting for the centered response wy, of shape (n,) or (n, q).

    density is the (weighted) marginal density on the grid. With sample weights,
    wy is already multiplied by them and K_proj = K * weights; otherwise
    K_proj = K. Returns the components on the grid, shape (d, ngrid[, q]).
    """
    d, ngrid, n = K.shape
    K_proj = K if K_proj is None else K_proj
    gw = grid_w if wy.ndim == 1 else grid_w[:, None]
    inv = 1.0 / density
    if wy.ndim == 2:
        inv = inv[:, :, None]
    center = np.empty_like(density)
    for j in range(d):
        mass = grid_w * density[j]
        center[j] = mass / mass.sum()

    # marginal local-constant fits, centered
    marginal = np.empty((d, ngrid) + wy.shape[1:])
    for j in range(d):
        c = K[j] @ wy / n * inv[j]
        marginal[j] = c - center[j] @ c

    # Gauss-Seidel sweeps; *_s holds each component smoothed at the sample points
    prev = marginal
    prev_s = np.array([K[j].T @ (gw * prev[j]) for j in range(d)])
    for _ in range(max_iter):
        curr = np.empty_like(prev)
        curr_s = np.empty_like(prev_s)
        updated = np.zeros_like(prev_s[0])
        pending = prev_s.sum(axis=0)
        for j in range(d):
            pending -= prev_s[j]
            m_j = marginal[j] - inv[j] * (K_proj[j] @ (updated + pending)) / n
            m_j -= center[j] @ m_j
            curr[j] = m_j
            curr_s[j] = K[j].T @ (gw * m_j)
            updated += curr_s[j]
        if np.max(np.sum((curr - prev) ** 2 * gw, axis=1)) < tol:
            break
        prev, prev_s = curr, curr_s
    return curr


def fit_sbf(X, y, h, ngrid):
    _, grid_w, K = boundary_kernel(X, h, ngrid)
    density = np.maximum(K.sum(axis=2) / len(X), 1e-8)
    c = y.mean()
    return c, backfit(K, grid_w, density, y - c)


def predict_sbf(c, components, X_new):
    t = np.linspace(0.0, 1.0, components.shape[1])
    pred = np.full(len(X_new), c)
    for j in range(X_new.shape[1]):
        pred += np.interp(np.clip(X_new[:, j], 0.0, 1.0), t, components[j])
    return pred


# Linear projection of an additive fit over the unlabeled points

def interpolation_moments(sample, ngrid):
    """M[j, k, g] = mean over U of X1_k times the interpolation weight of grid
    point g at X_j, so that mean(X1 * m) = c mean(X1) + sum_j M[j] @ m_j."""
    X_u, X_u1 = sample.X_u, sample.X_u1
    N, d = X_u.shape
    M = np.zeros((d, d + 1, ngrid))
    for j in range(d):
        pos = np.clip(X_u[:, j], 0.0, 1.0) * (ngrid - 1)
        left = np.minimum(pos.astype(np.intp), ngrid - 2)
        frac = pos - left
        for k in range(d + 1):
            np.add.at(M[j, k], left, X_u1[:, k] / N * (1.0 - frac))
            np.add.at(M[j, k], left + 1, X_u1[:, k] / N * frac)
    return M


def additive_beta(sample, M, c, components):
    moment = c * sample.X_u1.mean(axis=0)
    for j in range(len(components)):
        moment += M[j] @ components[j]
    return solve(sample.Gamma_u, moment)


def run(sample):
    X, y, ngrid = sample.X, sample.y, sample.ngrid
    X1 = add_intercept(X)
    M = interpolation_moments(sample, ngrid)
    grid = bandwidth_grid(len(X))
    scores = []
    for h in grid:
        fold_mse = []
        for train, val in sample.folds:
            c, comps = fit_sbf(X[train], y[train], h, ngrid)
            err = y[val] - X1[val] @ additive_beta(sample, M, c, comps)
            if not np.all(np.isfinite(err)):
                fold_mse = [np.inf]
                break
            fold_mse.append(np.mean(err**2))
        scores.append(np.mean(fold_mse))
    best = int(np.nanargmin(scores))
    c, comps = fit_sbf(X, y, grid[best], ngrid)
    return {"m_u": predict_sbf(c, comps, sample.X_u), "tuning": {"h": grid[best]},
            "cv_score": scores[best]}
