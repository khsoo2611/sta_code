"""RBF kernel ridge regression.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
sample : Sample
    Data and CV folds used to score the projected coefficients.

Methods
-------
fit selects q and the df fraction by CV, recalculating gamma and ridge on
each training fold and then on the full labeled sample.
The ridge grid is based on effective degrees of freedom.
predict gives a regression prediction for each new observation.

After fit, cv_scores has axes (q, df_fraction); cv_fold_scores and
cv_fold_ridges have axes (fold, q, df_fraction); cv_fold_gammas has axes
(fold, q). Fold order matches sample.folds. cv_fold_median_sq_dist stores D_med^2,
the square of the median Euclidean distance, not the median squared distance.
cv_selected_indices gives the winning (q, df) indices.
"""

import numpy as np

from utils import (
    add_intercept, eigh_psd, krr_fit, median_distance, ridge_for_df,
    solve, sq_dist,
)

RBF_Q = np.geomspace(0.02, 0.85, 10)
RBF_DF_FRACTION = np.geomspace(0.025, 0.75, 10)


class KRR_R:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self, sample):
        """Select the kernel scale and ridge value using sample.folds."""
        folds = sample.folds
        X, y = self.X, self.y
        n = len(X)
        X1 = add_intercept(X)
        scores = np.zeros((len(RBF_Q), len(RBF_DF_FRACTION)))
        fold_medians = np.empty(len(folds))
        fold_gammas = np.empty((len(folds), len(RBF_Q)))
        fold_ridges = np.empty((len(folds),) + scores.shape)
        fold_scores = np.empty_like(fold_ridges)
        for fold_idx, (train, val) in enumerate(folds):
            X_train = X[train]
            fold_medians[fold_idx] = median_distance(X_train) ** 2
            gammas = -np.log(RBF_Q) / fold_medians[fold_idx]
            fold_gammas[fold_idx] = gammas
            D2_train = sq_dist(X_train, X_train)
            D2_eval = sq_dist(sample.X_u, X_train)
            for g_idx, g in enumerate(gammas):
                K_train = np.exp(-g * D2_train)
                K_eval = np.exp(-g * D2_eval)
                lam, V = eigh_psd(K_train)
                ridges = ridge_for_df(lam, RBF_DF_FRACTION * train.size)
                fold_ridges[fold_idx, g_idx] = ridges
                alpha = V @ ((V.T @ y[train])[:, None]
                             / (lam[:, None] + ridges[None, :]))
                moment = sample.X_u1.T @ K_eval / len(sample.X_u) @ alpha
                beta = solve(sample.Gamma_u, moment)
                err = X1[val] @ beta - y[val, None]
                finite = np.isfinite(err)
                counts = finite.sum(axis=0)
                sums = np.sum(np.where(finite, err * err, 0.0), axis=0)
                score = np.divide(
                    sums, counts, out=np.full(len(ridges), np.inf),
                    where=counts == len(val),
                )
                fold_scores[fold_idx, g_idx] = score
                scores[g_idx] += score
        scores /= len(folds)
        i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
        gamma = -np.log(RBF_Q[i]) / median_distance(X) ** 2
        K = np.exp(-gamma * sq_dist(X, X))
        lam, _ = eigh_psd(K)
        ridge = ridge_for_df(lam, [RBF_DF_FRACTION[j] * n])[0]
        self.tuning = {"q": RBF_Q[i], "gamma": gamma,
                       "df_fraction": RBF_DF_FRACTION[j], "ridge": ridge}
        self.cv_score = scores[i, j]
        self.cv_scores = scores
        self.cv_fold_median_sq_dist = fold_medians
        self.cv_fold_gammas = fold_gammas
        self.cv_fold_ridges = fold_ridges
        self.cv_fold_scores = fold_scores
        self.cv_selected_indices = (i, j)
        self.alpha = krr_fit(K, y, self.tuning["ridge"])
        return self

    def predict(self, X_new):
        K_new = np.exp(-self.tuning["gamma"] * sq_dist(X_new, self.X))
        return K_new @ self.alpha
