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

Gamma_w uses labeled and unlabeled covariates to compute the weights.
The final projection uses Gamma_u from unlabeled covariates only.

Methods
-------
fit estimates the weighted additive components and constraint coefficients.
predict gives a regression prediction for each new observation.
"""

import numpy as np

from utils import (
    add_intercept, additive_beta, backfit, sbf_bandwidth_grid, boundary_kernel,
    interpolation_moments, solve,
)


class ATA_CSBF:
    def __init__(self, X, y, ngrid=101):
        self.X = X
        self.y = y
        self.ngrid = ngrid

    def fit(self, h=None, *, sample=None, weights=None):
        X, y, ngrid = self.X, self.y, self.ngrid
        X1 = add_intercept(X)
        self.cv_score = None
        if h is None:
            A = solve(sample.Gamma_w, add_intercept(X).T).T
            weights = np.sum(A * A, axis=1)
            M = interpolation_moments(sample, ngrid)
            grid = sbf_bandwidth_grid(len(X))
            scores = []
            for h in grid:
                fold_mse = []
                for train, val in sample.folds:
                    fitted = ATA_CSBF(X[train], y[train], ngrid).fit(
                        h=h, weights=weights[train],
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

        self.h = h
        self.weights = weights
        self.tuning = {"h": h}
        w = self.weights
        n, d = X.shape
        _, grid_w, K = boundary_kernel(X, h, ngrid)
        density = np.maximum(K @ w / n, 1e-10)
        Y = np.column_stack([y, X1 / w[:, None]])
        c = w @ Y / (n * np.mean(w))
        comps = backfit(K, grid_w, density, w[:, None] * (Y - c), K * w)

        fitted = np.tile(c, (n, 1))
        for j in range(d):
            fitted += K[j].T @ (grid_w[:, None] * comps[j])
        base, basis = fitted[:, 0], fitted[:, 1:]
        B = X1.T @ basis / n
        lam = solve(B, X1.T @ (y - base) / n)
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
