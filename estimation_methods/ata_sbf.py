"""ATA-SBF: smooth backfitting on [0, 1].

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
    Labeled data, unlabeled covariates, and CV folds used to select h.

Methods
-------
fit estimates the additive components.
predict gives a regression prediction for each new observation.
"""

import numpy as np

from utils import (
    add_intercept, additive_beta, backfit, sbf_bandwidth_grid, boundary_kernel,
    interpolation_moments,
)


class ATA_SBF:
    def __init__(self, X, y, ngrid=101):
        self.X = X
        self.y = y
        self.ngrid = ngrid

    def fit(self, h=None, *, sample=None):
        X, y, ngrid = self.X, self.y, self.ngrid
        self.cv_score = None
        if h is None:
            X1 = add_intercept(X)
            M = interpolation_moments(sample, ngrid)
            grid = sbf_bandwidth_grid(len(X))
            scores = []
            for h in grid:
                fold_mse = []
                for train, val in sample.folds:
                    fitted = ATA_SBF(X[train], y[train], ngrid).fit(h=h)
                    err = y[val] - X1[val] @ additive_beta(
                        sample, M, fitted.c, fitted.components,
                    )
                    if not np.all(np.isfinite(err)):
                        fold_mse = [np.inf]
                        break
                    fold_mse.append(np.mean(err**2))
                scores.append(np.mean(fold_mse))
            best = int(np.nanargmin(scores))
            h = grid[best]
            self.cv_score = scores[best]

        self.h = h
        self.tuning = {"h": h}
        _, grid_w, K = boundary_kernel(X, h, ngrid)
        density = np.maximum(K.sum(axis=2) / len(X), 1e-8)
        self.c = y.mean()
        self.components = backfit(K, grid_w, density, y - self.c)
        return self

    def predict(self, X_new):
        t = np.linspace(0.0, 1.0, self.components.shape[1])
        pred = np.full(len(X_new), self.c)
        for j in range(X_new.shape[1]):
            pred += np.interp(np.clip(X_new[:, j], 0.0, 1.0), t, self.components[j])
        return pred
