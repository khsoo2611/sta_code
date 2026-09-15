"""LP-NW and LP-LL: full-dimensional Nadaraya-Watson and local-linear imputation.

Both use a product kernel with one bandwidth for all coordinates. A query point
without enough kernel support is imputed as 0; the share of such points is
reported as fallback_fraction.
"""

import numpy as np

from .utils import add_intercept, n1_5_grid, solve, theory_grid

BATCH = 512  # query points per block of kernel weights
TINY = 1e-12


def epanechnikov_weights(U):
    return np.prod(0.75 * np.maximum(1.0 - U**2, 0.0), axis=-1)


def gaussian_weights(U):
    # scaled so that the largest weight of each query point is one
    e = -0.5 * np.sum(U**2, axis=-1)
    return np.exp(e - e.max(axis=-1, keepdims=True))


KERNELS = {"epanechnikov": epanechnikov_weights, "gaussian": gaussian_weights}


def predict_nw(X, y, X_new, h, kernel="epanechnikov"):
    weight = KERNELS[kernel]
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
    return pred, empty.mean()  # share of points imputed as 0


def predict_ll(X, y, X_new, h, kernel="epanechnikov"):
    # A point is imputed as 0 when fewer than d + 1 training points get positive
    # weight or when its local design matrix is rank deficient.
    weight = KERNELS[kernel]
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

        full = np.linalg.matrix_rank(A, tol=1e-10) == d + 1
        if full.any():
            try:
                coef = np.linalg.solve(A[full], b[full][:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                coef = np.array([solve(a, v) for a, v in zip(A[full], b[full])])
            rows = s + np.flatnonzero(ok)[full]
            pred[rows] = coef[:, 0]
            failed[rows] = False
    return pred, failed.mean()


def select_bandwidth(predict, sample, grid):
    # beta-based CV: impute U from the training folds, project onto (1, X_u)
    # and score X beta_hat on the validation fold
    X, y = sample.X, sample.y
    X1 = add_intercept(X)
    scores = []
    for h in grid:
        fold_mse = []
        for train, val in sample.folds:
            m_u, _ = predict(X[train], y[train], sample.X_u, h)
            err = y[val] - X1[val] @ sample.project(m_u)
            fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
        scores.append(np.mean(fold_mse))
    best = int(np.nanargmin(scores))
    return grid[best], scores[best]


def run(sample, smoother, grid):
    """smoother is "nw" or "ll"; grid is "n1_5" (simulations) or "theory" (real data)."""
    predict = predict_nw if smoother == "nw" else predict_ll
    n, d = sample.X.shape
    hs = n1_5_grid(n) if grid == "n1_5" else theory_grid(n, d)
    h, cv_score = select_bandwidth(predict, sample, hs)
    m_u, fallback = predict(sample.X, sample.y, sample.X_u, h)
    return {"m_u": m_u, "tuning": {"h": h}, "cv_score": cv_score, "fallback_fraction": fallback}
