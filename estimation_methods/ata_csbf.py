"""ATA-CSBF: weighted smooth backfitting with a calibration step.

Labeled point i gets weight w_i = ||Gamma_w^{-1} (1, X_i)||^2. The response and
the columns of X1 / w are smoothed together, and m = m_y + m_basis @ lambda
with lambda chosen so that mean(X1 * (y - m)) = 0 holds for the smoothed
fitted values at the labeled points.
"""

import numpy as np

from .ata_sbf import additive_beta, backfit, bandwidth_grid, boundary_kernel, interpolation_moments
from .utils import add_intercept, solve


def csbf_weights(X, Gamma_w):
    A = solve(Gamma_w, add_intercept(X).T).T
    return np.sum(A * A, axis=1)


def fit_csbf(X, y, w, h, ngrid):
    # column 0 of c and comps is for y, the other d + 1 columns for the basis X1 / w
    n, d = X.shape
    X1 = add_intercept(X)
    _, grid_w, K = boundary_kernel(X, h, ngrid)
    density = np.maximum(K @ w / n, 1e-10)
    Y = np.column_stack([y, X1 / w[:, None]])
    c = w @ Y / (n * np.mean(w))
    comps = backfit(K, grid_w, density, w[:, None] * (Y - c), K * w)

    # smoothed fitted values at the labeled points, then the calibration
    fitted = np.tile(c, (n, 1))
    for j in range(d):
        fitted += K[j].T @ (grid_w[:, None] * comps[j])
    base, basis = fitted[:, 0], fitted[:, 1:]
    B = X1.T @ basis / n
    lam = solve(B, X1.T @ (y - base) / n)
    calibration = X1.T @ (y - (base + basis @ lam)) / n
    diagnostics = {
        "weight_min": float(np.min(w)),
        "weight_median": float(np.median(w)),
        "weight_max": float(np.max(w)),
        "moment_condition": float(np.linalg.cond(B)),
        "calibration_max_abs": float(np.max(np.abs(calibration))),
    }
    return c, comps, lam, diagnostics


def interpolate(c, components, X_new):
    ngrid = components.shape[1]
    out = np.tile(c, (len(X_new), 1))
    for j in range(X_new.shape[1]):
        pos = np.clip(X_new[:, j], 0.0, 1.0) * (ngrid - 1)
        left = np.minimum(pos.astype(np.intp), ngrid - 2)
        frac = (pos - left)[:, None]
        out += components[j, left] * (1.0 - frac) + components[j, left + 1] * frac
    return out


def run(sample):
    X, y, ngrid = sample.X, sample.y, sample.ngrid
    X1 = add_intercept(X)
    w = csbf_weights(X, sample.Gamma_w)
    M = interpolation_moments(sample, ngrid)
    grid = bandwidth_grid(len(X))
    scores = []
    for h in grid:
        fold_mse = []
        for train, val in sample.folds:
            c, comps, lam, _ = fit_csbf(X[train], y[train], w[train], h, ngrid)
            c_m = c[0] + c[1:] @ lam
            comps_m = comps[:, :, 0] + np.einsum("dgp,p->dg", comps[:, :, 1:], lam, optimize=True)
            err = y[val] - X1[val] @ additive_beta(sample, M, c_m, comps_m)
            fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
        scores.append(np.mean(fold_mse))
    best = int(np.nanargmin(scores))
    c, comps, lam, diagnostics = fit_csbf(X, y, w, grid[best], ngrid)
    fitted = interpolate(c, comps, sample.X_u)
    return {"m_u": fitted[:, 0] + fitted[:, 1:] @ lam, "tuning": {"h": grid[best]},
            "cv_score": scores[best], **diagnostics}
