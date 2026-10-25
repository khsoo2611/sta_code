"""PI: the partial-information estimator of Azriel et al. (2022).

Parameters
----------
sample : Sample
    Labeled covariates X, responses y, and unlabeled covariates X_u.

Methods
-------
fit uses covariate moments from the labeled and unlabeled observations.
beta stores the intercept followed by the slope coefficients.
"""

import numpy as np

from utils import solve


class PI:
    def __init__(self, sample):
        self.sample = sample

    def fit(self):
        sample = self.sample
        X, y = sample.X, sample.y
        Z = np.vstack([X, sample.X_u])  # moments of X use L and U together
        n, d = X.shape
        slopes = np.empty(d)
        for j in range(d):
            keep = [k for k in range(d) if k != j]
            # Regress X_j on the other covariates and an intercept.
            # Use labeled and unlabeled data to compute the residual variance.
            A = np.column_stack([np.ones(len(Z)), Z[:, keep]])
            coef = solve(A.T @ A / len(Z), A.T @ Z[:, j] / len(Z))
            e_all = Z[:, j] - A @ coef
            s2 = np.mean(e_all * e_all)
            e = X[:, j] - np.column_stack([np.ones(n), X[:, keep]]) @ coef  # the same residual on L

            # beta_j = E(y e) / s2. The terms e / s2 and X_k e / s2 - 1{k = j} have
            # mean zero, so regressing on them leaves the intercept as a
            # variance-reduced estimate of beta_j.
            controls = X * e[:, None] / s2
            controls[:, j] -= 1.0
            D = np.column_stack([np.ones(n), e / s2, controls])
            slopes[j] = np.linalg.lstsq(D, y * e / s2, rcond=None)[0][0]
        intercept = np.mean(y) - slopes @ np.mean(X, axis=0)
        self.beta = np.concatenate([[intercept], slopes])
        return self
