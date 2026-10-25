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
    One bandwidth for all variables. None selects it by cross-validation (CV).
sample : Sample
    Data and CV folds used to select h.
grid : str
    "mode_1" uses the simulation bandwidth grid.
    "mode_2" uses the real-data bandwidth grid.
    The formulas are in utils.mode_1_grid and utils.mode_2_grid.
sparse_sums : bool
    M1 LL sums over support points one at a time in sparse regions.

Methods
-------
fit selects or sets the bandwidth.
predict gives a regression prediction for each new observation.
When a local fit fails, its prediction is zero.
fallback_fraction records the proportion of these predictions.
"""

from functools import partial

import numpy as np

from utils import (
    KERNELS,
    mode_1_grid,
    select_lp_bandwidth,
    solve,
    mode_2_grid,
)

BATCH = 512  # query points per block of kernel weights
TINY = 1e-12


class LL:
    def __init__(self, X, y, kernel="epanechnikov", *, sparse_sums=False):
        self.X = X
        self.y = y
        self.kernel = kernel
        self._sparse_sums = sparse_sums

    def fit(self, h=None, *, sample=None, grid="mode_1"):
        if sample is not None:
            self._sparse_sums = (
                sample.info.get("model") == "M1"
                and self.kernel == "epanechnikov"
                and grid == "mode_1"
            )
        if h is None:
            n, d = self.X.shape
            hs = mode_1_grid(n) if grid == "mode_1" else mode_2_grid(n, d)
            regression = partial(LL, sparse_sums=True) if self._sparse_sums else LL
            project = None
            if self._sparse_sums:
                gamma_inv = np.linalg.pinv(sample.Gamma_u)
                project = lambda m: gamma_inv @ (sample.X_u1.T @ m / len(m))
            self.h, self.cv_score = select_lp_bandwidth(
                regression, self.kernel, sample, hs, project=project,
            )
        else:
            self.h = h
            self.cv_score = None
        self.tuning = {"h": self.h}
        return self

    def predict(self, X_new):
        # A point is imputed as 0 when fewer than d + 1 training points get positive
        # weight or when its local design matrix is rank deficient.
        X, y, h = self.X, self.y, self.h
        weight = KERNELS[self.kernel]
        d = X.shape[1]
        if self._sparse_sums and self.kernel == "epanechnikov" and d >= 7 and len(X_new):
            sample_idx = np.linspace(0, len(X_new) - 1, min(32, len(X_new)), dtype=int)
            within_support = np.all(
                np.abs(X[None, :, :] - X_new[sample_idx, None, :]) < h,
                axis=2,
            )
            if float(np.mean(within_support)) < 0.25:
                return self._predict_sparse(X_new)
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

            full = np.linalg.matrix_rank(A, tol=1e-10) == d + 1
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

    def _predict_sparse(self, X_new):
        # Sum over support points in their original row order.
        X, y, h = self.X, self.y, self.h
        q = X.shape[1] + 1
        pred = np.zeros(len(X_new))
        failed = np.ones(len(X_new), dtype=bool)
        for row, x in enumerate(X_new):
            indices = np.flatnonzero(np.all(np.abs(X - x) < h, axis=1))
            if indices.size < q:
                continue
            U = (X[indices] - X_new[row]) / h
            w = 1.0 - U * U
            w[np.abs(U) >= 1.0] = 0.0
            w *= 0.75
            w = np.prod(w, axis=-1)
            active = w > TINY
            if (
                not np.isfinite(w).all()
                or float(np.sum(w)) <= TINY
                or int(np.sum(active)) < q
            ):
                continue
            indices = indices[active]
            U = U[active]
            w = w[active]
            wU = w[:, None] * U
            A = np.empty((q, q), dtype=float)
            A[0, 0] = float(np.sum(w))
            A[0, 1:] = np.sum(wU, axis=0)
            A[1:, 0] = A[0, 1:]
            A[1:, 1:] = U.T @ wU
            if np.linalg.matrix_rank(A, tol=1e-10) < q:
                continue
            wy = w * y[indices]
            b = np.empty(q, dtype=float)
            b[0] = float(np.sum(wy))
            b[1:] = U.T @ wy
            pred[row] = float(solve(A, b)[0])
            failed[row] = False
        self.fallback_fraction = failed.mean()
        return pred
