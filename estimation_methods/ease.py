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
    Score the final EASE coefficients on held-out observations.

Methods and Functions
---------------------
fit computes the EASE coefficients in beta.
make_folds returns cross-fitting groups and SS-SIR-projected X values.
sir_directions computes the SS-SIR directions.
_fit_cross_fitted fits one candidate across all cross-fitting groups.
_combine_with_ols combines the SNP and OLS coefficients.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .nw import NW
from .ll import LL
from .krr_m import MATERN_Q, MATERN_GAMMA_FRACTION, matern, matern_cv, matern_ell
from .krr_r import RBF_Q, RBF_DF_FRACTION
from utils import (
    Sample, add_intercept, eigh_psd, krr_fit, median_distance, median_sq_dist,
    ridge_for_df, solve, sq_dist,
)


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

def make_folds(sample, K, r, H):
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
        folds.append({
            "train": train,
            "X_train": X_train,
            "X_hold": X_hold,
            "X_u": X_u,
        })
    return blocks, folds


class EASE:
    """EASE using the regression method given by smoother."""

    def __init__(self, sample, smoother, *, K, r, H, cv_folds):
        self.sample = sample
        self.smoother = smoother
        self.K = K
        self.r = r
        self.H = H
        self.cv_folds = cv_folds

    def _fit_cross_fitted(self, candidate):
        self.m_hold, m_u_folds, self.tuning = [], [], []
        for fold in self.folds:
            pred_hold, pred_u, fold_tuning = self._fit_fixed_regression(fold, candidate)
            self.m_hold.append(pred_hold)
            m_u_folds.append(pred_u)
            self.tuning.append(fold_tuning)

        sample = self.sample
        fixed = self._fixed_design if self.smoother == "matern" else None
        if fixed is None:
            X1_blocks = [add_intercept(sample.X[b]) for b in self.blocks]
            G = [X1.T @ X1 for X1 in X1_blocks]
        else:
            X1_blocks, G = fixed["X1_blocks"], fixed["G"]
        R = [X1.T @ (sample.y[b] - m).reshape(-1, 1)
             for X1, b, m in zip(X1_blocks, self.blocks, self.m_hold)]
        eta = solve(sum(G) if fixed is None else fixed["G_total"], sum(R))
        self.eta = eta
        m_avg = np.stack([m.reshape(-1, 1) for m in m_u_folds]).mean(axis=0)
        mu_u = m_avg + (sample.X_u1 @ eta).reshape(-1, 1)
        self.beta_snp = solve(sample.Gamma_u, sample.X_u1.T @ mu_u / len(sample.X_u))
        self.m_u = mu_u.reshape(-1)
        self.beta = self._combine_with_ols(X1_blocks, G, R, fixed)

    def _prepare_fixed_design(self, in_cv=False):
        """Compute values shared by all candidates."""
        sample = self.sample
        X1_blocks = [add_intercept(sample.X[b]) for b in self.blocks]
        G = [X1.T @ X1 for X1 in X1_blocks]
        G_total = sum(G)
        X1 = add_intercept(sample.X)
        beta_ols = solve(X1.T @ X1, X1.T @ sample.y).reshape(-1, 1)
        psi_ols = []
        for k, b in enumerate(self.blocks):
            res_ols = sample.y[b].reshape(-1, 1) - X1_blocks[k] @ beta_ols
            psi_ols.append(solve(sample.Gamma_u, X1_blocks[k].T * res_ols.reshape(1, -1)))
        self._fixed_design = {
            "X1_blocks": X1_blocks, "G": G, "G_total": G_total,
            "G_leave": [G_total - G[k] for k in range(len(G))],
            "beta_ols": beta_ols, "psi_ols": psi_ols,
            "in_cv": in_cv,
        }

    def _tuning_candidates(self):
        if self.smoother in ("nw", "ll"):
            return [{"bandwidth_factor": float(c)} for c in np.geomspace(0.05, 1.0, 20)]
        if self.smoother == "matern":
            return [{"q": float(q), "gamma_fraction": float(f)}
                    for q in MATERN_Q for f in MATERN_GAMMA_FRACTION]
        return [{"q": float(q), "df_fraction": float(f)}
                for q in RBF_Q for f in RBF_DF_FRACTION]

    def fit(self):
        sample = self.sample
        if len(sample.folds) != self.cv_folds:
            raise ValueError("EASE CV fold count does not match cv_folds")
        candidates = self._tuning_candidates()
        scores = np.empty((len(sample.folds), len(candidates)))
        fold_betas, fold_tuning = [], []
        X1 = add_intercept(sample.X)
        for fold_idx, (train, val) in enumerate(sample.folds):
            training_sample = Sample(
                seed=sample.seed, X=sample.X[train], y=sample.y[train],
                X_u=sample.X_u, beta0=np.empty(0), folds=[],
            )
            fitted = EASE(
                training_sample, self.smoother, K=self.K, r=self.r, H=self.H,
                cv_folds=self.cv_folds,
            )
            fitted.blocks, fitted.folds = make_folds(
                training_sample, self.K, self.r, self.H,
            )
            if self.smoother == "matern":
                fitted._prepare_fixed_design(in_cv=True)
            betas, tuning = [], []
            for candidate_idx, candidate in enumerate(candidates):
                fitted._fit_cross_fitted(candidate)
                beta = np.reshape(fitted.beta, -1)
                err = sample.y[val] - X1[val] @ beta
                scores[fold_idx, candidate_idx] = (
                    np.mean(err**2) if np.all(np.isfinite(err)) else np.inf
                )
                betas.append(beta.copy())
                tuning.append([dict(t) for t in fitted.tuning])
            fold_betas.append(betas)
            fold_tuning.append(tuning)
            del fitted

        mean_scores = scores.mean(axis=0)
        if not np.isfinite(mean_scores).any():
            raise ValueError("no finite final EASE CV score")
        best = int(np.argmin(mean_scores))
        self.blocks, self.folds = make_folds(sample, self.K, self.r, self.H)
        if self.smoother == "matern":
            self._prepare_fixed_design()
        self._fit_cross_fitted(candidates[best])
        if not np.all(np.isfinite(self.beta)):
            raise ValueError("non-finite final EASE coefficients")
        self.tuning_grid = candidates
        self.selected_candidate_index = best
        self.selected_candidate = dict(candidates[best])
        self.cv_scores = mean_scores
        self.cv_fold_scores = scores
        self.cv_score = float(mean_scores[best])
        self.cv_fold_betas = np.array([b[best] for b in fold_betas])
        self.cv_fold_tuning = [t[best] for t in fold_tuning]
        # Axes are validation fold, candidate, and EASE cross-fitting block.
        self.cv_all_fold_tuning = {
            key: np.array([[[t[key] for t in candidate] for candidate in fold]
                           for fold in fold_tuning])
            for key in fold_tuning[0][0][0]
        }
        self.cv_validation_indices = [val.copy() for _, val in sample.folds]
        return self

    def _fit_fixed_regression(self, fold, candidate):
        """Fit one tuning candidate on one cross-fitting group."""
        X_train = fold["X_train"]
        data = fold.get("_fixed_matern_data") if self.smoother == "matern" else None
        if data is None:
            y_train = self.sample.y[fold["train"]]
            X_new = np.vstack([fold["X_hold"], fold["X_u"]])
            if self.smoother == "matern":
                data = {
                    "y_train": y_train, "d_med": np.sqrt(median_sq_dist(X_train)),
                    "distance": np.sqrt(sq_dist(X_train, X_train)),
                    "distance_new": np.sqrt(sq_dist(X_new, X_train)),
                }
                if self._fixed_design["in_cv"]:
                    fold["_fixed_matern_data"] = data
        else:
            y_train = data["y_train"]
        n, d = X_train.shape
        if self.smoother == "matern":
            t = d // 2 + 1
        if self.smoother in ("nw", "ll"):
            h = candidate["bandwidth_factor"] * n ** (-1 / (d + 4))
            smoother = NW if self.smoother == "nw" else LL
            model = smoother(X_train, y_train, kernel="gaussian").fit(h=h)
            pred = model.predict(X_new)
            tuning = {**candidate, "h": h, "n_train": n,
                      "fallback_fraction": model.fallback_fraction}
        else:
            # Consecutive ridge candidates share the same kernel matrices.
            kernel = fold.get("_fixed_kernel")
            if kernel is None or kernel["q"] != candidate["q"]:
                if self.smoother == "matern":
                    # Match the distance scale in KRR_M.fit.
                    d_med = data["d_med"]
                    nu = t - d / 2
                    scale = matern_ell(d_med, nu, candidate["q"])
                    # Use interpolation only during CV.
                    # Use direct values for the final fit.
                    kernel_function = matern_cv if self._fixed_design["in_cv"] else matern
                    K = kernel_function(data["distance"], scale, nu)
                    K_new = kernel_function(data["distance_new"], scale, nu)
                else:
                    d_med = median_distance(X_train)
                    scale = -np.log(candidate["q"]) / d_med**2
                    K = np.exp(-scale * sq_dist(X_train, X_train))
                    K_new = np.exp(-scale * sq_dist(X_new, X_train))
                if self.smoother == "matern" and self._fixed_design["in_cv"]:
                    fractions = [float(f) for f in MATERN_GAMMA_FRACTION]
                    ridges = [n * n ** (-f * 2 * t / d) for f in fractions]
                    predictions = self._matern_ridge_predictions(K, K_new, y_train, ridges)
                    kernel = {"q": candidate["q"], "scale": scale,
                              "predictions": dict(zip(fractions, predictions.T.copy()))}
                else:
                    kernel = {"q": candidate["q"], "scale": scale, "K": K, "K_new": K_new}
                if self.smoother == "rbf":
                    kernel["eigvals"] = eigh_psd(K)[0]
                fold["_fixed_kernel"] = kernel
            if self.smoother == "matern":
                ridge = n * n ** (-candidate["gamma_fraction"] * 2 * t / d)
                tuning = {**candidate, "ell": kernel["scale"], "ridge": ridge, "n_train": n}
            else:
                ridge = ridge_for_df(kernel["eigvals"], [candidate["df_fraction"] * n])[0]
                tuning = {**candidate, "gamma": kernel["scale"], "ridge": ridge, "n_train": n}
            if "predictions" in kernel:
                pred = kernel["predictions"][candidate["gamma_fraction"]]
            else:
                pred = kernel["K_new"] @ krr_fit(kernel["K"], y_train, ridge)
        k = len(fold["X_hold"])
        return pred[:k], pred[k:], tuning

    def _matern_ridge_predictions(self, K, K_new, y, ridges):
        """Reuse the kernel eigendecomposition across CV ridge candidates."""
        try:
            values, vectors = np.linalg.eigh(K, UPLO="U")
            shifted = values[:, None] + np.asarray(ridges)
            if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0.0):
                raise np.linalg.LinAlgError("non-positive shifted kernel eigenvalue")
            alpha = vectors @ ((vectors.T @ y)[:, None] / shifted)
            if not np.all(np.isfinite(alpha)):
                raise np.linalg.LinAlgError("non-finite ridge coefficients")
        except np.linalg.LinAlgError:
            # If this fails, solve each ridge system separately.
            alpha = np.column_stack([krr_fit(K, y, ridge) for ridge in ridges])
        return K_new @ alpha

    def _combine_with_ols(self, X1_blocks, G, R, fixed):
        """Return beta_ols + Delta (beta_snp - beta_ols), with diagonal Delta.

        For diff = psi_snp - psi_ols, each diagonal entry is
        -mean(diff_j * psi_ols_j) / (mean(diff_j**2) + eps).
        These means use all labeled observations without centering.
        Each block uses an SNP correction fitted without that block.
        """
        sample = self.sample
        X, y = sample.X, sample.y
        if fixed is None:
            X1 = add_intercept(X)
            beta_ols = solve(X1.T @ X1, X1.T @ y).reshape(-1, 1)
        else:
            beta_ols = fixed["beta_ols"]
        p = len(beta_ols)
        cross, square = np.zeros((p, p)), np.zeros((p, p))
        for k, b in enumerate(self.blocks):
            eta_k = solve(sum(G) - G[k] if fixed is None else fixed["G_leave"][k],
                          sum(R) - R[k])
            res_snp = y[b].reshape(-1, 1) - (self.m_hold[k].reshape(-1, 1) + X1_blocks[k] @ eta_k)
            psi_snp = solve(sample.Gamma_u, X1_blocks[k].T * res_snp.reshape(1, -1))
            if fixed is None:
                res_ols = y[b].reshape(-1, 1) - X1_blocks[k] @ beta_ols
                psi_ols = solve(sample.Gamma_u, X1_blocks[k].T * res_ols.reshape(1, -1))
            else:
                psi_ols = fixed["psi_ols"][k]
            diff = psi_snp - psi_ols
            cross += diff @ psi_ols.T
            square += diff @ diff.T
        n = len(X)
        eps = n ** -0.25 / np.log(n)
        delta = np.diag((-np.diag(cross) / n) / (np.diag(square) / n + eps))
        self.delta = delta
        return beta_ols + delta @ (self.beta_snp.reshape(-1, 1) - beta_ols)
