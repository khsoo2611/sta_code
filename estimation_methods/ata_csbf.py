"""ATA-CSBF: weighted and constrained smooth backfitting.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
ngrid : int
    Number of points used to represent each additive component.
h : float or None
    Bandwidth. None selects it by cross-validation (CV).
sample : Sample
    Data and CV folds used to select h and compute the sample weights.
weights : array (n,)
    Training sample weights supplied when h is fixed.
kernel : tuple or None
    (grid_w, K) from normalized_kernel(X, h, ngrid), when already computed.
    CV computes it once for all labeled points and passes the training columns.
diagnostics : bool
    Compute diagnostic values. Internal CV fits skip them.

The weights and final projection use Gamma_u from the N unlabeled observations.

Methods
-------
fit estimates the weighted additive components and constraint coefficients.
predict gives a regression prediction for each new observation.
"""

import numpy as np

from utils import (
    add_intercept, additive_beta, backfit, sbf_bandwidth_grid, normalized_kernel,
    interpolation_moments, solve,
)


class ATA_CSBF:
    def __init__(self, X, y, ngrid=101):
        self.X = X
        self.y = y
        self.ngrid = ngrid

    def fit(self, h=None, *, sample=None, weights=None, kernel=None, diagnostics=True):
        X, y, ngrid = self.X, self.y, self.ngrid
        X1 = add_intercept(X)
        if h is None:
            A = solve(sample.Gamma_u, X1.T).T
            weights = np.sum(A * A, axis=1)
            M = interpolation_moments(sample, ngrid)
            grid = sbf_bandwidth_grid(len(X))
            scores = []
            for h in grid:
                # Each column of K depends only on its own observation, so the
                # training folds take columns of one computation.
                grid_w, K_all = normalized_kernel(X, h, ngrid)
                fold_mse = []
                for train, val in sample.folds:
                    fitted = ATA_CSBF(X[train], y[train]).fit(
                        h=h, weights=weights[train], kernel=(grid_w, K_all[:, :, train]),
                        diagnostics=False,
                    )
                    c, comps, lam = fitted.c, fitted.components, fitted.lam
                    c_m = c[0] + c[1:] @ lam
                    comps_m = comps[:, :, 0] + np.einsum(
                        "dgp,p->dg", comps[:, :, 1:], lam, optimize=True,
                    )
                    err = y[val] - X1[val] @ additive_beta(sample, M, c_m, comps_m)
                    fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
                scores.append(np.mean(fold_mse))
            best = int(np.nanargmin(scores))
            h = grid[best]
            self.cv_score = scores[best]
            self.tuning = {"h": h}

        w = weights
        n, d = X.shape
        grid_w, K = kernel if kernel is not None else normalized_kernel(X, h, ngrid)
        density = np.maximum(K @ w / n, 1e-10)
        # Fit y and the constraint functions together with the same smoother.
        Y = np.column_stack([y, X1 / w[:, None]])
        c = w @ Y / (n * np.mean(w))
        comps = backfit(K, grid_w, density, w[:, None] * (Y - c), K * w)

        fitted = np.tile(c, (n, 1))
        for j in range(d):
            fitted += K[j].T @ (grid_w[:, None] * comps[j])
        base, basis = fitted[:, 0], fitted[:, 1:]
        B = X1.T @ basis / n
        # Solve the sample moment constraints.
        lam = solve(B, X1.T @ (y - base) / n)
        if diagnostics:
            calibration = X1.T @ (y - (base + basis @ lam)) / n
            self.diagnostics = {
                "weight_min": float(np.min(w)),
                "weight_median": float(np.median(w)),
                "weight_max": float(np.max(w)),
                "moment_condition": float(np.linalg.cond(B)),
                "calibration_max_abs": float(np.max(np.abs(calibration))),
            }
        self.c, self.components, self.lam = c, comps, lam
        return self

    def predict(self, X_new):
        ngrid = self.components.shape[1]
        fitted = np.tile(self.c, (len(X_new), 1))
        for j in range(X_new.shape[1]):
            pos = np.clip(X_new[:, j], 0.0, 1.0) * (ngrid - 1)
            left = np.minimum(pos.astype(np.intp), ngrid - 2)
            frac = (pos - left)[:, None]
            fitted += self.components[j, left] * (1.0 - frac) + self.components[j, left + 1] * frac
        return fitted[:, 0] + fitted[:, 1:] @ self.lam
