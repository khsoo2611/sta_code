"""Nadaraya-Watson (NW) regression.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
kernel : str
    Direct methods use the Epanechnikov kernel.
    EASE uses the Gaussian kernel.
h : float or None
    One bandwidth for all variables. None selects it by cross-validation (CV)
    with the Epanechnikov kernel.
    The default candidates are in utils.lp_bandwidth_grid, capped at 0.8.
sample : Sample
    Data and CV folds used to select h.

Methods
-------
fit selects or sets the bandwidth.
predict gives a regression prediction for each new observation.
When a local fit fails, its prediction is zero.
fallback_fraction records the proportion of these predictions.
cv_predictions gives the CV predictions for every bandwidth and fold at once.
"""

import numpy as np

from utils import (
    KERNELS,
    epanechnikov_weights,
    fold_order,
    leave_one_out_sums,
    lp_bandwidth_grid,
    select_lp_bandwidth,
)

BATCH = 512  # query points per block of kernel weights
TINY = 1e-12


class NW:
    def __init__(self, X, y, kernel="epanechnikov"):
        self.X = X
        self.y = y
        self.kernel = kernel

    def fit(self, h=None, *, sample=None):
        if h is None:
            if self.kernel != "epanechnikov":
                raise ValueError("Bandwidth CV requires the Epanechnikov kernel")
            n, d = self.X.shape
            hs = lp_bandwidth_grid(n, d)
            self.h, self.cv_score = select_lp_bandwidth(NW, sample, hs)
            self.tuning = {"h": self.h}
        else:
            self.h = h
        return self

    def predict(self, X_new):
        X, y, h = self.X, self.y, self.h
        weight = KERNELS[self.kernel]
        pred = np.zeros(len(X_new))
        empty = np.zeros(len(X_new), dtype=bool)
        for s in range(0, len(X_new), BATCH):
            W = weight((X[None, :, :] - X_new[s:s + BATCH, None, :]) / h)
            total = W.sum(axis=1)
            ok = (total > TINY) & np.any(W > TINY, axis=1)
            block = np.zeros(len(W))
            block[ok] = (W @ y)[ok] / total[ok]
            pred[s:s + BATCH] = block
            empty[s:s + BATCH] = ~ok
        self.fallback_fraction = empty.mean()  # share of points imputed as 0
        return pred

    @staticmethod
    def cv_predictions(X, y, grid, folds, X_new):
        """Predictions on X_new for every bandwidth in grid and every fold.

        pred[i, k] is NW(X[train], y[train]).fit(h=grid[i]).predict(X_new) for
        fold k, with the Epanechnikov kernel, up to rounding. The kernel weights
        are computed once for all labeled points.
        """
        order, bounds = fold_order(folds, len(X))
        X, y = X[order], y[order]
        pred = np.zeros((len(grid), len(folds), len(X_new)))
        for s in range(0, len(X_new), BATCH):
            D = X[None, :, :] - X_new[s:s + BATCH, None, :]
            U, factors = np.empty_like(D), np.empty_like(D)
            for i, h in enumerate(grid):
                np.divide(D, h, out=U)
                W = epanechnikov_weights(U, out=factors)
                parts = [
                    (W[:, a:b].sum(axis=1), W[:, a:b] @ y[a:b], (W[:, a:b] > TINY).sum(axis=1))
                    for a, b in zip(bounds[:-1], bounds[1:])
                ]
                for k, (total, wy, active) in enumerate(leave_one_out_sums(parts)):
                    ok = (total > TINY) & (active > 0)
                    pred[i, k, s + np.flatnonzero(ok)] = wy[ok] / total[ok]
        return pred
