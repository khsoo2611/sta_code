"""RBF kernel ridge regression.

Parameters
----------
X : array (n, d)
    Labeled covariates, with n observations and d variables.
y : array (n,)
    Responses for the labeled observations.
folds : list
    Training and validation indices for cross-validation (CV).
sample : Sample or None
    Direct methods pass sample to use beta-based CV.
    EASE passes None to use regression prediction MSE.

Methods
-------
fit selects the kernel scale and ridge value by CV.
The ridge grid is based on effective degrees of freedom.
predict gives a regression prediction for each new observation.
"""

import numpy as np

from utils import (
    add_intercept, krr_cv_scores, eigh_psd, krr_fit, median_sq_dist, ridge_for_df,
    solve, sq_dist,
)

RBF_Q = np.geomspace(0.02, 0.85, 10)
RBF_DF_FRACTION = np.geomspace(0.025, 0.75, 10)


class KRR_R:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self, folds, sample=None):
        """Select the kernel scale and ridge value using the supplied CV folds."""
        X, y = self.X, self.y
        n = len(X)
        D2 = sq_dist(X, X)
        X1 = add_intercept(X)
        gammas = -np.log(RBF_Q) / median_sq_dist(X)
        df_ridges = lambda eigvals, n_train: ridge_for_df(eigvals, RBF_DF_FRACTION * n_train)
        if sample is None:
            scores = np.array([
                krr_cv_scores(np.exp(-g * D2), y, folds, df_ridges)
                for g in gammas
            ])
        else:
            # Direct KRR uses distances computed within each training fold.
            scores = np.zeros((len(gammas), len(RBF_DF_FRACTION)))
            for train, val in folds:
                X_train = X[train]
                D2_train = sq_dist(X_train, X_train)
                D2_u = sq_dist(sample.X_u, X_train)
                for g_idx, g in enumerate(gammas):
                    K_train = np.exp(-g * D2_train)
                    K_u = np.exp(-g * D2_u)
                    lam, V = eigh_psd(K_train)
                    ridges = np.maximum(df_ridges(lam, train.size), 1e-12)
                    alpha = V @ ((V.T @ y[train])[:, None]
                                 / (lam[:, None] + ridges[None, :]))
                    moment = sample.X_u1.T @ K_u / len(sample.X_u) @ alpha
                    beta = solve(sample.Gamma_u, moment)
                    err = X1[val] @ beta - y[val, None]
                    finite = np.isfinite(err)
                    counts = finite.sum(axis=0)
                    sums = np.sum(np.where(finite, err * err, 0.0), axis=0)
                    scores[g_idx] += np.divide(
                        sums, counts, out=np.full(len(ridges), np.inf),
                        where=counts == len(val),
                    )
            scores /= len(folds)
        i, j = np.unravel_index(np.nanargmin(scores), scores.shape)
        lam, _ = eigh_psd(np.exp(-gammas[i] * D2))
        ridge = ridge_for_df(lam, [RBF_DF_FRACTION[j] * n])[0]
        self.tuning = {"q": RBF_Q[i], "gamma": gammas[i],
                       "df_fraction": RBF_DF_FRACTION[j], "ridge": ridge}
        self.cv_score = scores[i, j]
        K = np.exp(-self.tuning["gamma"] * sq_dist(X, X))
        self.alpha = krr_fit(K, y, self.tuning["ridge"])
        return self

    def predict(self, X_new):
        K_new = np.exp(-self.tuning["gamma"] * sq_dist(X_new, self.X))
        return K_new @ self.alpha
