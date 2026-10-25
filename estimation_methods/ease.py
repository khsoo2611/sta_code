"""EASE with SS-SIR dimension reduction.

Parameters
----------
sample : Sample
    Labeled data, unlabeled covariates, and the random seed.
smoother : str
    Regression method: "nw", "ll", "matern", or "rbf".
K : int
    Number of cross-fitting groups in the labeled sample.
    Fit on K - 1 groups and predict the remaining group.
r : int
    Number of covariates after SS-SIR dimension reduction.
H : int
    Number of equal-width response intervals used by SS-SIR.
cv_folds : int
    Number of CV groups used to compare tuning candidates.
    CV uses only the K - 1 training groups from each cross-fitting step.
cache : dict
    Stores folds and SS-SIR results shared by the EASE methods.

Methods and Functions
---------------------
fit computes the EASE coefficients in beta.
make_folds builds the cross-fitting groups and tuning CV folds.
sir_directions computes the SS-SIR directions.
_fit_regression fits a regression for each cross-fitting group.
_combine_with_ols combines the SNP and OLS coefficients.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .nw import NW
from .ll import LL
from .krr_m import KRR_M
from .krr_r import KRR_R
from utils import add_intercept, kfold, solve


def ease_bandwidth_grid(n, d):
    """20 bandwidths c n^(-1/(d+4)), c geometric in [0.05, 1] for EASE fits."""
    return n ** (-1 / (d + 4)) * np.geomspace(0.05, 1.0, 20)


# Semi-supervised SIR

def slice_index(values, edges):
    if edges[-1] <= edges[0]:
        return np.zeros(len(values), dtype=np.intp)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def sir_directions(X, y, X_u, r, H):
    """Return the r SS-SIR directions as columns of a matrix.

    Split the response range into H equal-width intervals.
    Assign each unlabeled point to its nearest labeled neighbor's interval.
    Neighbors are found using standardized covariates.
    Compute each interval mean from labeled and unlabeled points together.
    Compute interval probabilities from labeled responses only.
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

def make_folds(sample, K, r, H, cv_folds):
    idx = np.arange(len(sample.X))
    np.random.default_rng(sample.seed).shuffle(idx)
    blocks = np.array_split(idx, K)
    folds = []
    for k, hold in enumerate(blocks):
        train = np.concatenate([b for i, b in enumerate(blocks) if i != k])
        X_train, X_hold, X_u = sample.X[train], sample.X[hold], sample.X_u
        # SIR directions are estimated without block k.
        P = sir_directions(X_train, sample.y[train], X_u, r, H)
        X_train = to_unit_cube(X_train, P)
        X_hold = to_unit_cube(X_hold, P)
        X_u = to_unit_cube(X_u, P)
        # The fixed value 9109 separates tuning CV seeds from other random seeds.
        inner_seed = int(np.random.SeedSequence([sample.seed, 9109, k]).generate_state(1)[0])
        folds.append({
            "train": train,
            "X_train": X_train,
            "X_hold": X_hold,
            "X_u": X_u,
            "inner_folds": kfold(len(train), cv_folds, inner_seed),
        })
    return blocks, folds


class EASE:
    """EASE using the regression method given by smoother."""

    def __init__(self, sample, smoother, *, K, r, H, cv_folds, cache):
        self.sample = sample
        self.smoother = smoother
        self.K = K
        self.r = r
        self.H = H
        self.cv_folds = cv_folds
        self.cache = cache

    def fit(self):
        if "folds" not in self.cache:
            self.cache["folds"] = make_folds(self.sample, self.K, self.r, self.H, self.cv_folds)
        self.blocks, self.folds = self.cache["folds"]

        self.m_hold, self.m_u_folds, self.tuning, self.cv_score = [], [], [], []
        for fold in self.folds:
            pred_hold, pred_u, fold_tuning, fold_score = self._fit_regression(fold)
            self.m_hold.append(pred_hold)
            self.m_u_folds.append(pred_u)
            self.tuning.append(fold_tuning)
            self.cv_score.append(fold_score)

        sample = self.sample
        X1_blocks = [add_intercept(sample.X[b]) for b in self.blocks]
        G = [X1.T @ X1 for X1 in X1_blocks]
        R = [X1.T @ (sample.y[b] - m).reshape(-1, 1)
             for X1, b, m in zip(X1_blocks, self.blocks, self.m_hold)]
        eta = solve(sum(G), sum(R))
        m_avg = np.stack([m.reshape(-1, 1) for m in self.m_u_folds]).mean(axis=0)
        mu_u = m_avg + (sample.X_u1 @ eta).reshape(-1, 1)
        self.beta_snp = solve(sample.Gamma_u, sample.X_u1.T @ mu_u / len(sample.X_u))
        self.m_u = mu_u.reshape(-1)
        self.beta = self._combine_with_ols(X1_blocks, G, R)
        return self

    def _fit_regression(self, fold):
        """Fit on the training groups and predict the remaining group and X_u."""
        X_train, y_train = fold["X_train"], self.sample.y[fold["train"]]
        X_new = np.vstack([fold["X_hold"], fold["X_u"]])
        if self.smoother in ("nw", "ll"):
            smoother = NW if self.smoother == "nw" else LL
            grid = ease_bandwidth_grid(*X_train.shape)
            scores = []
            for h in grid:
                fold_mse = []
                for train, val in fold["inner_folds"]:
                    model = smoother(X_train[train], y_train[train], kernel="gaussian").fit(h=h)
                    err = y_train[val] - model.predict(X_train[val])
                    fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
                scores.append(np.mean(fold_mse))
            best = int(np.nanargmin(scores))
            h, score = grid[best], scores[best]
            model = smoother(X_train, y_train, kernel="gaussian").fit(h=h)
            pred = model.predict(X_new)
            tuning = {"h": h}
        else:
            smoother = KRR_M if self.smoother == "matern" else KRR_R
            model = smoother(X_train, y_train).fit(fold["inner_folds"], sample=None)
            pred = model.predict(X_new)
            tuning, score = model.tuning, model.cv_score
        k = len(fold["X_hold"])
        return pred[:k], pred[k:], tuning, score

    def _combine_with_ols(self, X1_blocks, G, R):
        """Return beta_ols + Delta (beta_snp - beta_ols), with diagonal Delta.

        For diff = psi_snp - psi_ols, each diagonal entry is
        -mean(diff_j * psi_ols_j) / (mean(diff_j**2) + eps).
        These means use all labeled observations without centering.
        Each block uses an SNP correction fitted without that block.
        """
        sample = self.sample
        X, y = sample.X, sample.y
        X1 = add_intercept(X)
        beta_ols = solve(X1.T @ X1, X1.T @ y).reshape(-1, 1)
        p = X1.shape[1]
        cross, square = np.zeros((p, p)), np.zeros((p, p))
        for k, b in enumerate(self.blocks):
            eta_k = solve(sum(G) - G[k], sum(R) - R[k])
            res_snp = y[b].reshape(-1, 1) - (self.m_hold[k].reshape(-1, 1) + X1_blocks[k] @ eta_k)
            res_ols = y[b].reshape(-1, 1) - X1_blocks[k] @ beta_ols
            psi_snp = solve(sample.Gamma_u, X1_blocks[k].T * res_snp.reshape(1, -1))
            psi_ols = solve(sample.Gamma_u, X1_blocks[k].T * res_ols.reshape(1, -1))
            diff = psi_snp - psi_ols
            cross += diff @ psi_ols.T
            square += diff @ diff.T
        n = len(X)
        eps = n ** -0.25 / np.log(n)
        delta = np.diag((-np.diag(cross) / n) / (np.diag(square) / n + eps))
        return beta_ols + delta @ (self.beta_snp.reshape(-1, 1) - beta_ols)
