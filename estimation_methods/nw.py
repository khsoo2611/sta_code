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
    One bandwidth for all variables. None selects it by cross-validation (CV).
sample : Sample
    Data and CV folds used to select h.
grid : str
    "mode_1" uses the simulation bandwidth grid.
    "mode_2" uses the real-data bandwidth grid.
    The formulas are in utils.mode_1_grid and utils.mode_2_grid.

Methods
-------
fit selects or sets the bandwidth.
predict gives a regression prediction for each new observation.
When a local fit fails, its prediction is zero.
fallback_fraction records the proportion of these predictions.
"""

import numpy as np

from utils import (
    KERNELS,
    mode_1_grid,
    select_lp_bandwidth,
    mode_2_grid,
)

BATCH = 512  # query points per block of kernel weights
TINY = 1e-12


class NW:
    def __init__(self, X, y, kernel="epanechnikov"):
        self.X = X
        self.y = y
        self.kernel = kernel

    def fit(self, h=None, *, sample=None, grid="mode_1"):
        if h is None:
            n, d = self.X.shape
            hs = mode_1_grid(n) if grid == "mode_1" else mode_2_grid(n, d)
            self.h, self.cv_score = select_lp_bandwidth(NW, self.kernel, sample, hs)
        else:
            self.h = h
            self.cv_score = None
        self.tuning = {"h": self.h}
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
