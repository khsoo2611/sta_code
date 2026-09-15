"""SNP and EASE with cross-fitted nuisance regressions.

The labeled sample is split into K blocks. For each block the nuisance
regression is fitted on the other blocks and evaluated on the block and on U,
either on the full X or on r semi-supervised SIR directions found inside the
fold. SNP projects the residual-corrected imputations onto the linear span;
EASE combines SNP with OLS coordinate-wise.

In the paper, EASE(F) is SNP on the full X and EASE is the SIR version
combined with OLS.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import krr, lp
from .utils import add_intercept, kfold, optimal_grid, solve


# Semi-supervised SIR

def slice_index(values, edges):
    if edges[-1] <= edges[0]:
        return np.zeros(len(values), dtype=np.intp)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def sir_directions(X, y, X_u, r, H):
    """Leading r SIR directions of X, as a (p, r) matrix.

    Slice means pool the labeled points and the unlabeled points, the latter
    taking the response of their nearest labeled neighbor in standardized
    coordinates; slice probabilities come from the labeled responses.
    """
    p = X.shape[1]
    X_all = np.vstack([X, X_u])
    mu = X_all.mean(axis=0)
    Sigma = np.cov(X_all - mu, rowvar=False, bias=False) + 1e-8 * np.eye(p)
    lam, V = np.linalg.eigh(Sigma)
    root_inv = V @ np.diag(1.0 / np.sqrt(np.clip(lam, 1e-8, None))) @ V.T
    Z, Z_u = (X - mu) @ root_inv, (X_u - mu) @ root_inv

    edges = np.linspace(y.min(), y.max(), H + 1)
    s = slice_index(y, edges)
    nearest = NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(Z_u, return_distance=False)
    s_u = slice_index(y[nearest.reshape(-1)], edges)
    p_h = np.bincount(s, minlength=H) / len(y)

    M = np.zeros((p, p))
    for h in range(H):
        Z_h = np.vstack([Z[s == h], Z_u[s_u == h]])
        m = Z_h.mean(axis=0).reshape(-1, 1) if len(Z_h) else np.zeros((p, 1))
        M += p_h[h] * (m @ m.T)
    vals, vecs = np.linalg.eigh(M)
    return root_inv @ vecs[:, np.argsort(vals)[::-1][:r]]


def to_unit_cube(X, P):
    """X @ P rescaled by the range of x @ P over x in [0, 1]^p."""
    lo = np.minimum(P, 0.0).sum(axis=0)
    hi = np.maximum(P, 0.0).sum(axis=0)
    return np.clip((X @ P - lo) / (hi - lo), 0.0, 1.0)


# Cross-fitting

def make_folds(sample, K, r, H, cv_folds, sir):
    idx = np.arange(len(sample.X))
    np.random.default_rng(sample.seed).shuffle(idx)
    blocks = np.array_split(idx, K)
    folds = []
    for k, hold in enumerate(blocks):
        train = np.concatenate([b for i, b in enumerate(blocks) if i != k])
        X_train, X_hold, X_u = sample.X[train], sample.X[hold], sample.X_u
        if sir:
            # the nuisance regression uses r SIR directions estimated without block k
            P = sir_directions(X_train, sample.y[train], X_u, r, H)
            X_train = to_unit_cube(X_train, P)
            X_hold = to_unit_cube(X_hold, P)
            X_u = to_unit_cube(X_u, P)
        # 9109 is just a fixed tag so that the inner CV splits get their own random stream
        inner_seed = int(np.random.SeedSequence([sample.seed, 9109, k]).generate_state(1)[0])
        folds.append({
            "train": train,
            "X_train": X_train,
            "X_hold": X_hold,
            "X_u": X_u,
            "inner_folds": kfold(len(train), cv_folds, inner_seed),
        })
    return blocks, folds


def local_bandwidth(predict, X, y, folds):
    """Bandwidth from the optimal-rate grid by held-out prediction error."""
    grid = optimal_grid(*X.shape)
    scores = []
    for h in grid:
        fold_mse = []
        for train, val in folds:
            err = y[val] - predict(X[train], y[train], X[val], h, kernel="gaussian")[0]
            fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
        scores.append(np.mean(fold_mse))
    best = int(np.nanargmin(scores))
    return grid[best], scores[best]


def nuisance_fit(smoother, fold, y):
    """Fit on the fold's training part; predict at its held-out block and at U."""
    X_train, y_train = fold["X_train"], y[fold["train"]]
    X_new = np.vstack([fold["X_hold"], fold["X_u"]])
    if smoother in ("nw", "ll"):
        predict = lp.predict_nw if smoother == "nw" else lp.predict_ll
        h, score = local_bandwidth(predict, X_train, y_train, fold["inner_folds"])
        pred = predict(X_train, y_train, X_new, h, kernel="gaussian")[0]
        tuning = {"h": h}
    else:
        pred, tuning, score = krr.impute(X_train, y_train, X_new, fold["inner_folds"], smoother)
    k = len(fold["X_hold"])
    return pred[:k], pred[k:], tuning, score


# SNP and the EASE combination

def residual_moments(sample, blocks, m_hold):
    """Per block: X1_k, X1_k'X1_k and X1_k'(y_k - m_hat_k)."""
    X1_blocks = [add_intercept(sample.X[b]) for b in blocks]
    G = [X1.T @ X1 for X1 in X1_blocks]
    R = [X1.T @ (sample.y[b] - m).reshape(-1, 1) for X1, b, m in zip(X1_blocks, blocks, m_hold)]
    return X1_blocks, G, R


def fit_snp(sample, smoother, blocks, folds):
    m_hold, m_u, tuning, cv_score = [], [], [], []
    for fold in folds:
        pred_hold, pred_u, fold_tuning, fold_score = nuisance_fit(smoother, fold, sample.y)
        m_hold.append(pred_hold)
        m_u.append(pred_u)
        tuning.append(fold_tuning)
        cv_score.append(fold_score)

    # Residual correction: regress y - m_hat on (1, X) over all blocks, then
    # add the fitted correction to the average of the K fits on U.
    _, G, R = residual_moments(sample, blocks, m_hold)
    eta = solve(sum(G), sum(R))
    m_avg = np.stack([m.reshape(-1, 1) for m in m_u]).mean(axis=0)
    mu_u = m_avg + (sample.X_u1 @ eta).reshape(-1, 1)
    beta = solve(sample.Gamma_u, sample.X_u1.T @ mu_u / len(sample.X_u))
    return {"beta": beta, "m_u": mu_u.reshape(-1), "m_hold": m_hold,
            "tuning": tuning, "cv_score": cv_score}


def combine_with_ols(sample, blocks, snp):
    """beta_ols + Delta (beta_snp - beta_ols), Delta diagonal.

    Delta_jj = -Cov(psi_ols, psi_snp - psi_ols) / (Var(psi_snp - psi_ols) + eps),
    from the influence terms of both estimators on each block, where the SNP
    residual correction is refitted without that block.
    """
    X, y = sample.X, sample.y
    X1 = add_intercept(X)
    beta_ols = solve(X1.T @ X1, X1.T @ y).reshape(-1, 1)
    X1_blocks, G, R = residual_moments(sample, blocks, snp["m_hold"])
    p = X1.shape[1]
    cross, square = np.zeros((p, p)), np.zeros((p, p))
    for k, b in enumerate(blocks):
        eta_k = solve(sum(G) - G[k], sum(R) - R[k])
        res_snp = y[b].reshape(-1, 1) - (snp["m_hold"][k].reshape(-1, 1) + X1_blocks[k] @ eta_k)
        res_ols = y[b].reshape(-1, 1) - X1_blocks[k] @ beta_ols
        psi_snp = solve(sample.Gamma_u, X1_blocks[k].T * res_snp.reshape(1, -1))
        psi_ols = solve(sample.Gamma_u, X1_blocks[k].T * res_ols.reshape(1, -1))
        diff = psi_snp - psi_ols
        cross += diff @ psi_ols.T
        square += diff @ diff.T
    n = len(X)
    eps = n ** -0.25 / np.log(n)
    delta = np.diag((-np.diag(cross) / n) / (np.diag(square) / n + eps))
    return beta_ols + delta @ (snp["beta"].reshape(-1, 1) - beta_ols)


def run(sample, smoother, sir, combine, *, K, r, H, cv_folds, cache):
    """SNP, or EASE when combine is True, with nuisance fits on SIR directions or the full X.

    smoother is "nw", "ll", "matern" or "rbf". Folds and SNP fits are cached per
    replication, so ease_nw and snp_nw_low fit the nuisance regressions only once.
    """
    if ("folds", sir) not in cache:
        cache["folds", sir] = make_folds(sample, K, r, H, cv_folds, sir)
    blocks, folds = cache["folds", sir]
    if ("snp", smoother, sir) not in cache:
        cache["snp", smoother, sir] = fit_snp(sample, smoother, blocks, folds)
    snp = cache["snp", smoother, sir]
    beta = combine_with_ols(sample, blocks, snp) if combine else snp["beta"]
    return {"beta": beta, "m_u": snp["m_u"], "tuning": snp["tuning"], "cv_score": snp["cv_score"]}
