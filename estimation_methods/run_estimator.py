from dataclasses import dataclass, field
from functools import partial

import numpy as np

from . import ata_csbf, ata_sbf, ease, krr, lp, pi
from .utils import add_intercept, solve, sq_error

# Each function takes a Sample and returns a dict with either "m_u" (imputed
# responses on U, projected below) or "beta", plus tuning information.
DIRECT = {
    "ata_sbf": ata_sbf.run,
    "ata_csbf": ata_csbf.run,
    "lp_nw_n1_5": partial(lp.run, smoother="nw", grid="n1_5"),
    "lp_nw_theory": partial(lp.run, smoother="nw", grid="theory"),
    "lp_ll_n1_5": partial(lp.run, smoother="ll", grid="n1_5"),
    "lp_ll_theory": partial(lp.run, smoother="ll", grid="theory"),
    "krr_m": partial(krr.run, kind="matern"),
    "krr_r": partial(krr.run, kind="rbf"),
    "pi": pi.run,
}
# SNP/EASE variants: (nuisance smoother, reduce X by SIR, combine with OLS).
# The paper uses ease_nw, ease_ll, ease_m and ease_r, with SIR and OLS combination.
# Other variants remain available for the earlier additional comparisons.
EASE = {
    "ease_f_nw": ("nw", False, False),
    "ease_f_ll": ("ll", False, False),
    "ease_f_m": ("matern", False, False),
    "ease_f_r": ("rbf", False, False),
    "ease_nw": ("nw", True, True),
    "ease_ll": ("ll", True, True),
    "ease_m": ("matern", True, True),
    "ease_r": ("rbf", True, True),
    "ease_nw_full": ("nw", False, True),
    "ease_ll_full": ("ll", False, True),
    "ease_matern_krr_full": ("matern", False, True),
    "ease_rbf_krr_full": ("rbf", False, True),
    "snp_nw_low": ("nw", True, False),
    "snp_ll_low": ("ll", True, False),
    "snp_matern_krr_low": ("matern", True, False),
    "snp_rbf_krr_low": ("rbf", True, False),
}
ESTIMATORS = [*DIRECT, *EASE]


@dataclass
class Sample:
    """Labeled data (X, y) and unlabeled covariates X_u of one replication.

    X_u1 is X_u with an intercept column and Gamma_u = X_u1'X_u1 / N; Gamma_w is
    the same Gram matrix over L and U together (ATA-CSBF weights). folds are the
    (train, validation) pairs shared by the CV of the direct estimators.
    """

    seed: int
    X: np.ndarray
    y: np.ndarray
    X_u: np.ndarray
    beta0: np.ndarray
    folds: list
    m_true: np.ndarray = None  # true regression function on U (simulations only)
    info: dict = field(default_factory=dict)  # copied into every result record
    ngrid: int = 101  # grid size of the additive components

    def __post_init__(self):
        self.X_u1 = add_intercept(self.X_u)
        self.Gamma_u = self.X_u1.T @ self.X_u1 / len(self.X_u)
        X1_all = add_intercept(np.vstack([self.X, self.X_u]))
        self.Gamma_w = X1_all.T @ X1_all / len(X1_all)

    def project(self, m_u):
        """Coefficients of the linear projection of m_u on (1, X_u)."""
        return solve(self.Gamma_u, self.X_u1.T @ m_u / len(self.X_u))


def fit_estimator(sample, name, *, K, r, H, cv_folds, cache):
    # K, r, H and cv_folds are only used by SNP/EASE (blocks, SIR directions,
    # slices, inner CV folds); `cache` is shared within a replication.
    if name in EASE:
        smoother, sir, combine = EASE[name]
        out = ease.run(sample, smoother, sir, combine,
                       K=K, r=r, H=H, cv_folds=cv_folds, cache=cache)
    else:
        out = DIRECT[name](sample)

    m_u = out.pop("m_u", None)
    beta = out.pop("beta", None)
    if beta is None:
        beta = sample.project(m_u)
    beta = np.reshape(beta, -1)
    record = {
        **sample.info,
        "estimator_id": name,
        "beta_target": sample.beta0,
        "beta_estimate": beta,
        "coefficient_squared_error": sq_error(beta, sample.beta0),
        **out,
    }
    if m_u is not None and sample.m_true is not None:
        diff = m_u - sample.m_true
        moment = np.mean(sample.X_u1 * diff[:, None], axis=0)
        record["regression_mse"] = float(np.mean(diff**2))
        record["moment_squared_error"] = float(moment @ moment)
    return record
