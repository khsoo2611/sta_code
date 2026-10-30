"""Local linear (LL) regression.

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
    full_rank,
    leave_one_out_sums,
    lp_bandwidth_grid,
    select_lp_bandwidth,
    solve,
)

BATCH = 512  # query points per block of kernel weights
TINY = 1e-12  # Weights at or below this value are left out of the local fit.


class LL:
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
            self.h, self.cv_score = select_lp_bandwidth(
                LL, sample, hs,
            )
            self.tuning = {"h": self.h}
        else:
            self.h = h
        return self

    def predict(self, X_new):
        # Return 0 if fewer than d + 1 points have weight above TINY
        # or the local design matrix does not have full rank.
        X, y, h = self.X, self.y, self.h
        weight = KERNELS[self.kernel]
        d = X.shape[1]
        pred = np.zeros(len(X_new))
        failed = np.ones(len(X_new), dtype=bool)
        for s in range(0, len(X_new), BATCH):
            U = (X[None, :, :] - X_new[s:s + BATCH, None, :]) / h
            W = weight(U)
            active = W > TINY
            ok = np.isfinite(W).all(axis=1) & (W.sum(axis=1) > TINY) & (active.sum(axis=1) >= d + 1)
            if not ok.any():
                continue
            U_ok = U[ok]
            W_ok = np.where(active[ok], W[ok], 0.0)
            WU = W_ok[:, :, None] * U_ok
            Ut = U_ok.transpose(0, 2, 1)

            # weighted normal equations for the local basis (1, (X_i - x) / h)
            A = np.empty((len(U_ok), d + 1, d + 1))
            A[:, 0, 0] = W_ok.sum(axis=1)
            A[:, 0, 1:] = WU.sum(axis=1)
            A[:, 1:, 0] = A[:, 0, 1:]
            A[:, 1:, 1:] = Ut @ WU
            Wy = W_ok * y
            b = np.empty((len(U_ok), d + 1))
            b[:, 0] = Wy.sum(axis=1)
            b[:, 1:] = (Ut @ Wy[:, :, None])[:, :, 0]

            # Use a fast rank check, with an SVD check near the cutoff.
            full = full_rank(A)
            if full.any():
                try:
                    coef = np.linalg.solve(A[full], b[full][:, :, None])[:, :, 0]
                except np.linalg.LinAlgError:
                    coef = np.array([solve(a, v) for a, v in zip(A[full], b[full])])
                rows = s + np.flatnonzero(ok)[full]
                pred[rows] = coef[:, 0]
                failed[rows] = False
        self.fallback_fraction = failed.mean()
        return pred

    @staticmethod
    def cv_predictions(X, y, grid, folds, X_new):
        """Predictions on X_new for every bandwidth in grid and every fold.

        pred[i, k] is LL(X[train], y[train]).fit(h=grid[i]).predict(X_new) for
        fold k, with the Epanechnikov kernel, up to rounding. The kernel weights
        are computed once for all labeled points. Each training fit adds the
        local sums of the other validation folds.
        """
        order, bounds = fold_order(folds, len(X))
        X, y = X[order], y[order]
        d = X.shape[1]
        pred = np.zeros((len(grid), len(folds), len(X_new)))
        for s in range(0, len(X_new), BATCH):
            D = X[None, :, :] - X_new[s:s + BATCH, None, :]
            U, factors = np.empty_like(D), np.empty_like(D)
            for i, h in enumerate(grid):
                np.divide(D, h, out=U)
                W = epanechnikov_weights(U, out=factors)
                W_act = np.where(W > TINY, W, 0.0)
                # local sums over each validation fold
                parts = []
                for a, b in zip(bounds[:-1], bounds[1:]):
                    U_f, W_f = U[:, a:b], W_act[:, a:b]
                    WU = W_f[:, :, None] * U_f
                    Wy = W_f * y[a:b]
                    Ut = U_f.transpose(0, 2, 1)
                    parts.append((
                        W_f.sum(axis=1), WU.sum(axis=1), Ut @ WU,
                        Wy.sum(axis=1), (Ut @ Wy[:, :, None])[:, :, 0],
                        W[:, a:b].sum(axis=1), (W[:, a:b] > TINY).sum(axis=1),
                        (~np.isfinite(W[:, a:b])).sum(axis=1),
                    ))
                for k, sums in enumerate(leave_one_out_sums(parts)):
                    w_sum, wu_sum, wuu_sum, wy_sum, wuy_sum, total, active, nonfinite = sums
                    ok = (nonfinite == 0) & (total > TINY) & (active >= d + 1)
                    if not ok.any():
                        continue
                    A = np.empty((ok.sum(), d + 1, d + 1))
                    A[:, 0, 0] = w_sum[ok]
                    A[:, 0, 1:] = wu_sum[ok]
                    A[:, 1:, 0] = A[:, 0, 1:]
                    A[:, 1:, 1:] = wuu_sum[ok]
                    rhs = np.empty((ok.sum(), d + 1))
                    rhs[:, 0] = wy_sum[ok]
                    rhs[:, 1:] = wuy_sum[ok]
                    full = full_rank(A)
                    if full.any():
                        try:
                            coef = np.linalg.solve(A[full], rhs[full][:, :, None])[:, :, 0]
                        except np.linalg.LinAlgError:
                            coef = np.array([solve(a, v) for a, v in zip(A[full], rhs[full])])
                        pred[i, k, s + np.flatnonzero(ok)[full]] = coef[:, 0]
        return pred
