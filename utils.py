"""Common data, estimation, cross-validation (CV), and result functions.

Sample stores labeled data (X, y), unlabeled covariates X_u, and CV folds.
X has n observations and d covariates. X_u has N observations.
Gamma_u uses the N unlabeled observations for ATA-CSBF weights and projection.

fit_estimator fits one method and records its coefficient error.
For direct methods with tuning, CV uses the prediction error of the projected beta.
EASE scores its final linear coefficients on the same CV folds.
select_lp_bandwidth computes NW and LL CV predictions for all folds at once,
sharing the kernel weights between folds.

save_settings saves the run settings.
save_results saves summary.csv and records.pkl.
risk_ratio computes RE from the sums of coefficient squared errors.
ratio_table arranges RE values by dimension or labeled sample size.
read_reference reads the saved result tables.
read_runs reads results from completed runs.
export writes CSV and LaTeX tables.
"""

import argparse
import concurrent.futures as cf
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def solve(A, b):
    """np.linalg.solve, falling back to least squares when A is singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def ols_beta(X, y):
    X1 = add_intercept(X)
    return solve(X1.T @ X1, X1.T @ y)


def sq_error(beta, beta0):
    return float(np.linalg.norm(np.reshape(beta, -1) - beta0) ** 2)


def kfold(n, k, seed):
    """Random k-fold split of range(n) as (train, validation) index pairs."""
    perm = np.random.default_rng(seed).permutation(n)
    return [(np.delete(np.arange(n), val), val) for val in np.array_split(perm, k)]


def risk_ratio(records, by):
    """RE = sum of OLS squared errors / sum of the estimator's squared errors."""
    est = records["coefficient_squared_error"].astype(float)
    ols = records["coefficient_squared_error_ols"].astype(float)
    if not np.all(np.isfinite(est) & np.isfinite(ols)):
        raise ValueError("records contain non-finite coefficient errors")
    summary = (
        records.assign(est=est, ols=ols)
        .groupby(by)
        .agg(reps=("est", "size"), ols=("ols", "sum"), est=("est", "sum"))
        .reset_index()
    )
    summary["ratio"] = summary["ols"] / summary["est"]
    return summary


def new_output_dir(path):
    path = Path(path)
    candidate, i = path, 2
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.name}_{i}")
            i += 1


def save_settings(folder, settings):
    with open(folder / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, sort_keys=True)


def save_results(folder, summary, records):
    summary.to_csv(folder / "summary.csv", index=False)
    records.to_pickle(folder / "records.pkl")


@dataclass
class Sample:
    """Data for one repetition.

    X_u1 adds an intercept column to the unlabeled covariates.
    Gamma_u = X_u1.T @ X_u1 / N.
    Direct methods share the training and validation indices in folds.
    """

    seed: int
    X: np.ndarray
    y: np.ndarray
    X_u: np.ndarray
    beta0: np.ndarray
    folds: list
    m_true: np.ndarray = None  # true regression function on U (simulations only)
    info: dict = field(default_factory=dict)  # copied into every result record

    def __post_init__(self):
        self.X_u1 = add_intercept(self.X_u)
        self.Gamma_u = self.X_u1.T @ self.X_u1 / len(self.X_u)

    def project(self, m_u):
        """Coefficients of the linear projection of m_u on (1, X_u)."""
        return solve(self.Gamma_u, self.X_u1.T @ m_u / len(self.X_u))


def sbf_bandwidth_grid(n):
    """20 equally spaced bandwidths strictly inside (n^(-1/2), n^(-1/4))."""
    lo, hi = n ** -0.5, n ** -0.25
    return lo + np.arange(1, 21) / 21 * (hi - lo)


def trapezoid_weights(t):
    dt = np.diff(t)
    return (np.append(dt, 0.0) + np.append(0.0, dt)) / 2.0


def normalized_kernel(X, h, ngrid):
    """Epanechnikov kernel weights normalized over [0, 1].

    Divide K_h(t_g - X_ij) by its integral over [0, 1] for each observation.
    Return trapezoid integration weights and kernel weights K.
    The kernel integral is computed from its formula.
    """
    n, d = X.shape
    t = np.linspace(0.0, 1.0, ngrid)
    K = np.empty((d, ngrid, n))
    for j in range(d):
        x = X[:, j]
        u = (t[:, None] - x[None, :]) / h
        K[j] = 0.75 * (1.0 - u**2) * (np.abs(u) < 1.0) / h
        a = (np.maximum(0.0, x - h) - x) / h
        b = (np.minimum(1.0, x + h) - x) / h
        K[j] /= 0.75 * ((b - a) - (b**3 - a**3) / 3.0)
    return trapezoid_weights(t), K


def backfit(K, grid_w, density, wy, K_proj=None, max_iter=100, tol=1e-8):
    """Estimate centered additive components by smooth backfitting.

    density contains the marginal densities on the grid.
    For weighted fits, wy includes the sample weights and K_proj = K * weights.
    For unweighted fits, K_proj = K.
    The output has shape (d, ngrid) for one response.
    With several responses, the last axis contains the response components.
    """
    d, ngrid, n = K.shape
    K_proj = K if K_proj is None else K_proj
    gw = grid_w if wy.ndim == 1 else grid_w[:, None]
    inv = 1.0 / density
    if wy.ndim == 2:
        inv = inv[:, :, None]
    center = np.empty_like(density)
    for j in range(d):
        mass = grid_w * density[j]
        center[j] = mass / mass.sum()

    # marginal local-constant fits, centered
    marginal = np.empty((d, ngrid) + wy.shape[1:])
    for j in range(d):
        c = K[j] @ wy / n * inv[j]
        marginal[j] = c - center[j] @ c

    # Update one additive component at a time.
    # prev_s and curr_s hold their smoothed values at the sample points.
    prev = marginal
    prev_s = np.array([K[j].T @ (gw * prev[j]) for j in range(d)])
    for _ in range(max_iter):
        curr = np.empty_like(prev)
        curr_s = np.empty_like(prev_s)
        updated = np.zeros_like(prev_s[0])
        pending = prev_s.sum(axis=0)
        for j in range(d):
            pending -= prev_s[j]
            m_j = marginal[j] - inv[j] * (K_proj[j] @ (updated + pending)) / n
            m_j -= center[j] @ m_j
            curr[j] = m_j
            curr_s[j] = K[j].T @ (gw * m_j)
            updated += curr_s[j]
        if np.max(np.sum((curr - prev) ** 2 * gw, axis=1)) < tol:
            break
        prev, prev_s = curr, curr_s
    return curr


def interpolation_moments(sample, ngrid):
    """Grid weights for projecting interpolated additive components on X_u.

    M[j, k, g] averages X_u1[:, k] times the interpolation weight at grid point g
    for covariate j. Then mean(X_u1 * m) = c * mean(X_u1) + sum_j M[j] @ m_j.
    """
    X_u, X_u1 = sample.X_u, sample.X_u1
    N, d = X_u.shape
    M = np.zeros((d, d + 1, ngrid))
    for j in range(d):
        pos = np.clip(X_u[:, j], 0.0, 1.0) * (ngrid - 1)
        left = np.minimum(pos.astype(np.intp), ngrid - 2)
        frac = pos - left
        for k in range(d + 1):
            np.add.at(M[j, k], left, X_u1[:, k] / N * (1.0 - frac))
            np.add.at(M[j, k], left + 1, X_u1[:, k] / N * frac)
    return M


def additive_beta(sample, M, c, components):
    moment = c * sample.X_u1.mean(axis=0)
    for j in range(len(components)):
        moment += M[j] @ components[j]
    return solve(sample.Gamma_u, moment)


LP_BANDWIDTH_MAX = 0.8


def lp_bandwidth_grid(n, d):
    """Cap n^(-k / (21 d)), k = 1, ..., 20, at 0.8 and remove duplicates."""
    grid = n ** -(np.arange(1, 21) / (21 * d))
    return np.unique(np.minimum(grid, LP_BANDWIDTH_MAX))[::-1]


def epanechnikov_weights(U, out=None):
    """Product Epanechnikov weights. With out, the factors are computed in that buffer."""
    if out is None:
        return np.prod(0.75 * np.maximum(1.0 - U**2, 0.0), axis=-1)
    # the same operations in place, without temporary arrays
    np.multiply(U, U, out=out)
    np.subtract(1.0, out, out=out)
    np.maximum(out, 0.0, out=out)
    np.multiply(0.75, out, out=out)
    return np.prod(out, axis=-1)


def gaussian_weights(U):
    # scaled so that the largest weight of each query point is one
    e = -0.5 * np.sum(U**2, axis=-1)
    return np.exp(e - e.max(axis=-1, keepdims=True))


KERNELS = {"epanechnikov": epanechnikov_weights, "gaussian": gaussian_weights}


def fold_order(folds, n):
    """Labeled indices sorted by validation fold, and the fold boundaries.

    Require at least two folds whose validation sets partition range(n).
    Each training set must contain the remaining observations, as in kfold.
    """
    if len(folds) < 2:
        raise ValueError("At least two CV folds are required")
    order = np.concatenate([val for _, val in folds])
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("Validation folds must partition the labeled sample")
    for train, val in folds:
        if not np.array_equal(np.sort(train), np.setdiff1d(np.arange(n), val)):
            raise ValueError("Training indices must exclude the validation fold")
    return order, np.cumsum([0] + [len(val) for _, val in folds])


def leave_one_out_sums(parts):
    """For each fold, add the parts from all other folds.

    Use prefix and suffix sums to avoid subtracting nearly equal sums.
    Each part is a tuple of arrays.
    """
    def add(p, q):
        if p is None or q is None:
            return q if p is None else p
        return tuple(a + b for a, b in zip(p, q))

    prefix, suffix = [None], [None]
    for part in parts[:-1]:
        prefix.append(add(prefix[-1], part))
    for part in parts[:0:-1]:
        suffix.append(add(suffix[-1], part))
    return [add(p, q) for p, q in zip(prefix, suffix[::-1])]


def full_rank(A, tol=1e-10):
    """Use eigenvalues for a fast rank check of symmetric matrices.

    Use matrix_rank near the cutoff to check the result with SVD.
    """
    smallest = np.min(np.abs(np.linalg.eigvalsh(A)), axis=-1)
    full = smallest > tol
    near = (smallest > 0.1 * tol) & (smallest < 10.0 * tol)
    if near.any():
        full[near] = np.linalg.matrix_rank(A[near], tol=tol) == A.shape[-1]
    return full


def select_lp_bandwidth(regression, sample, grid):
    X, y = sample.X, sample.y
    X1 = add_intercept(X)
    preds = regression.cv_predictions(X, y, grid, sample.folds, sample.X_u)
    scores = []
    for i in range(len(grid)):
        fold_mse = []
        for k, (_, val) in enumerate(sample.folds):
            m_u = preds[i, k]
            err = y[val] - X1[val] @ sample.project(m_u)
            fold_mse.append(np.mean(err**2) if np.all(np.isfinite(err)) else np.inf)
        scores.append(np.mean(fold_mse))
    best = int(np.nanargmin(scores))
    return grid[best], scores[best]


def sq_dist(A, B):
    D2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def median_sq_dist(X):
    D2 = sq_dist(X, X)[np.triu_indices(len(X), k=1)]
    D2 = D2[np.isfinite(D2) & (D2 > 0.0)]
    return float(np.median(D2)) if D2.size else 1.0


def median_distance(X):
    """Median Euclidean distance between distinct labeled observations."""
    D2 = sq_dist(X, X)[np.triu_indices(len(X), k=1)]
    D2 = D2[np.isfinite(D2)]
    distance = float(np.median(np.sqrt(D2))) if D2.size else 0.0
    return distance if distance > 0.0 else 1.0


def ridge_for_df(eigvals, target_df):
    """Choose ridge values by bisection, with a lower bound of 1e-12.

    Use the lower bound when the target is not below the kernel rank.
    """
    lam = np.maximum(eigvals[np.isfinite(eigvals)], 0.0)
    target = np.asarray(target_df, dtype=float)
    ridge = np.full_like(target, 1e-12)  # used when the target is not below rank(K)
    active = target < np.sum(lam > 0.0)
    goal = target[active]
    df = lambda r: np.sum(lam[:, None] / (lam[:, None] + r[None, :]), axis=0)
    lo = np.zeros_like(goal)
    hi = np.full_like(goal, max(float(np.max(lam)), 1.0))
    while np.any(df(hi) > goal):
        hi[df(hi) > goal] *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        above = df(mid) > goal
        lo[above] = mid[above]
        hi[~above] = mid[~above]
    ridge[active] = np.maximum(hi, 1e-12)
    return ridge


# Cross-validation and fitting

def eigh_psd(K):
    lam, V = np.linalg.eigh(0.5 * (K + K.T))
    return np.maximum(lam, 0.0), V


def krr_cv_scores(K, ridge_grid, K_u, X1, sample):
    """Mean CV error for each ridge candidate.

    ridge_grid contains the ridge values shared by the training folds.
    Project predictions on X_u and score X1 @ beta on validation data.
    """
    y, folds = sample.y, sample.folds
    total = 0.0
    for train, val in folds:
        lam, V = eigh_psd(K[np.ix_(train, train)])
        ridges = np.maximum(ridge_grid, 1e-12)
        alpha = V @ ((V.T @ y[train])[:, None] / (lam[:, None] + ridges[None, :]))
        beta = solve(sample.Gamma_u, sample.X_u1.T @ K_u[:, train] / len(K_u) @ alpha)
        pred = X1[val] @ beta
        mse = np.mean((pred - y[val, None]) ** 2, axis=0)
        total = total + np.where(np.isfinite(mse), mse, np.inf)
    return total / len(folds)


def krr_fit(K, y, ridge):
    A = K + ridge * np.eye(len(K))
    try:
        return linalg.solve(A, y, assume_a="pos", check_finite=False)
    except np.linalg.LinAlgError:
        return solve(A, y)


def run_replications(tasks, run_one, jobs, initializer=None, initargs=()):
    records = []
    with tqdm(total=len(tasks), desc="replications", unit="rep", dynamic_ncols=True) as bar:
        if jobs <= 1:
            if initializer is not None:
                initializer(*initargs)
            for task in tasks:
                records.extend(run_one(task))
                bar.update()
        else:
            with cf.ProcessPoolExecutor(jobs, initializer=initializer, initargs=initargs) as pool:
                for future in cf.as_completed([pool.submit(run_one, task) for task in tasks]):
                    records.extend(future.result())
                    bar.update()
    # Drop None and NaN values from each record.
    rows = []
    for record in records:
        row = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


ESTIMATORS = [
    "ata_sbf", "ata_csbf", "lp_nw", "lp_ll",
    "krr_m", "krr_r", "pi", "ease_nw", "ease_ll", "ease_m", "ease_r",
]


def fit_estimator(sample, name, *, K, r, H, cv_folds):
    from estimation_methods.ata_csbf import ATA_CSBF
    from estimation_methods.ata_sbf import ATA_SBF
    from estimation_methods.ease import EASE
    from estimation_methods.krr_m import KRR_M
    from estimation_methods.krr_r import KRR_R, RBF_Q, RBF_DF_FRACTION
    from estimation_methods.nw import NW
    from estimation_methods.ll import LL
    from estimation_methods.pi import PI

    DIRECT = {
        "ata_sbf": ATA_SBF,
        "ata_csbf": ATA_CSBF,
        "lp_nw": NW,
        "lp_ll": LL,
        "krr_m": KRR_M,
        "krr_r": KRR_R,
    }
    EASE_SMOOTHERS = {
        "ease_nw": "nw",
        "ease_ll": "ll",
        "ease_m": "matern",
        "ease_r": "rbf",
    }

    m_u = None
    if name in EASE_SMOOTHERS:
        fitted = EASE(
            sample, EASE_SMOOTHERS[name], K=K, r=r, H=H,
            cv_folds=cv_folds,
        ).fit()
        beta, m_u = fitted.beta, fitted.m_u
    elif name == "pi":
        fitted = PI(sample).fit()
        beta = fitted.beta
    else:
        fitted = DIRECT[name](sample.X, sample.y).fit(sample=sample)
        m_u = fitted.predict(sample.X_u)
        beta = sample.project(m_u)
    beta = np.reshape(beta, -1)
    record = {
        **sample.info,
        "estimator_id": name,
        "beta_target": sample.beta0,
        "beta_estimate": beta,
        "coefficient_squared_error": sq_error(beta, sample.beta0),
    }
    if name != "pi":
        record["tuning"] = fitted.tuning
        record["cv_score"] = fitted.cv_score
    if name in EASE_SMOOTHERS:
        record["cv_target"] = "final_ease_beta"
        record.update(
            tuning_grid=fitted.tuning_grid,
            selected_candidate=fitted.selected_candidate,
            selected_candidate_index=fitted.selected_candidate_index,
            cv_scores=fitted.cv_scores,
            cv_fold_scores=fitted.cv_fold_scores,
            cv_fold_betas=fitted.cv_fold_betas,
            cv_fold_tuning=fitted.cv_fold_tuning,
            cv_all_fold_tuning=fitted.cv_all_fold_tuning,
            cv_validation_indices=fitted.cv_validation_indices,
            cv_fold_seed=sample.seed,
            ease_eta=fitted.eta,
            ease_delta=fitted.delta,
            ease_beta_snp=fitted.beta_snp,
            ease_H_policy="fixed_configured_H",
            ease_H=H,
        )
    elif name == "krr_r":
        record.update(
            tuning_grid={"q": RBF_Q.tolist(), "df_fraction": RBF_DF_FRACTION.tolist()},
            cv_scores=fitted.cv_scores,
            cv_fold_scores=fitted.cv_fold_scores,
            cv_fold_gammas=fitted.cv_fold_gammas,
            cv_fold_ridges=fitted.cv_fold_ridges,
            cv_fold_median_sq_dist=fitted.cv_fold_median_sq_dist,
            cv_selected_indices=fitted.cv_selected_indices,
            cv_fold_seed=sample.seed,
            cv_validation_indices=[val.copy() for _, val in sample.folds],
            kernel_scale_source="fold_training_sample",
        )
    if name in ("ata_sbf", "ata_csbf"):
        record["kernel_normalization"] = "exact"
    if name == "ata_csbf":
        record.update(fitted.diagnostics)
        record["weight_source"] = "unlabeled"
        record["weight_sample_size"] = len(sample.X_u)
    elif name.startswith("lp_"):
        record["fallback_fraction"] = fitted.fallback_fraction
        record["bandwidth_max"] = LP_BANDWIDTH_MAX
        record["tuning_grid"] = lp_bandwidth_grid(len(sample.X), sample.X.shape[1]).tolist()
    if m_u is not None and sample.m_true is not None:
        diff = m_u - sample.m_true
        moment = np.mean(sample.X_u1 * diff[:, None], axis=0)
        record["regression_mse"] = float(np.mean(diff**2))
        record["moment_squared_error"] = float(moment @ moment)
    return record



DIMENSIONS = list(range(2, 16))
SAMPLE_SIZES = list(range(100, 1001, 100))
METHODS = [
    "ATA-SBF", "ATA-CSBF", "NW", "LL", "KRR-M", "KRR-R",
    "EASE-NW", "EASE-LL", "EASE-M", "EASE-R", "PI",
]
TABLE_ROWS = {
    "M1": ["LL", "ATA-SBF", "ATA-CSBF"],
    "M2": ["KRR-M", "ATA-SBF", "ATA-CSBF"],
    "M3": METHODS,
    "M4": METHODS,
    "protein": METHODS,
    "miami": METHODS,
}
ESTIMATOR_LABELS = {
    "ata_sbf": "ATA-SBF", "ata_csbf": "ATA-CSBF", "lp_nw": "NW", "lp_ll": "LL",
    "krr_m": "KRR-M", "krr_r": "KRR-R", "pi": "PI",
    "ease_nw": "EASE-NW", "ease_ll": "EASE-LL", "ease_m": "EASE-M", "ease_r": "EASE-R",
}


def check_table(table, model, columns):
    rows = TABLE_ROWS[model]
    complete = set(rows).issubset(table.index) and set(table.columns) == set(columns)
    if not complete or not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"{model}: expected rows {rows} and columns {columns}")
    table = table.loc[rows, columns].astype(float)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"{model}: missing or non-finite RE values")
    table.index.name = "Procedure"
    return table


def ratio_table(summary, model, digits=None):
    data = summary.copy()
    names = set(data["estimator_id"])
    unsupported = sorted(names - ESTIMATOR_LABELS.keys())
    if unsupported:
        raise ValueError(f"unsupported estimator IDs: {unsupported}")
    data["Procedure"] = data["estimator_id"].map(ESTIMATOR_LABELS)
    column = "n" if model in ("protein", "miami") else "d"
    table = data.pivot(index="Procedure", columns=column, values="ratio")
    rows = [name for name in TABLE_ROWS.get(model, []) if name in table.index]
    rows += [name for name in table.index if name not in rows]
    table = table.loc[rows]
    return table.round(digits) if digits is not None else table


def read_reference():
    folder = ROOT / "results"
    tables = {}
    for model in TABLE_ROWS:
        real = model in ("protein", "miami")
        source = folder / ("realdata" if real else "simulation") / model
        n = None if real else 500
        table = pd.read_csv(source / f"{model}.csv", index_col="Procedure",
                            float_precision="round_trip")
        table.columns = table.columns.astype(int)
        tables[model, n] = check_table(table, model, SAMPLE_SIZES if n is None else DIMENSIONS)
    return tables


def read_runs(folders):
    rows = []
    for folder in map(Path, folders):
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        model = settings.get("model") or settings["dataset"]
        if model not in TABLE_ROWS:
            raise ValueError(f"{folder}: {model} is not in the main paper")
        n = settings["n"]
        real = model in ("protein", "miami")
        if (real and n not in SAMPLE_SIZES) or (not real and n != 500):
            raise ValueError(f"{folder}: n={n} is not in the main-paper table")
        paper = {"N": 5000, "reps": 100, "cv_folds": 10,
                 "seed": {"protein": 202607196, "miami": 202605310}.get(model, 1)}
        if model in ("M3", "M4", "protein", "miami"):
            paper.update(K=5, r=2, H=n // 5)
        if not real:
            paper["noise_sigma"] = 1.0
        wrong = {key: settings.get(key) for key in paper if settings.get(key) != paper[key]}
        if wrong:
            raise ValueError(f"{folder}: settings differ from the paper: {wrong}")

        summary = pd.read_csv(folder / "summary.csv", float_precision="round_trip")
        for rec in summary.to_dict(orient="records"):
            estimator_id = rec["estimator_id"]
            if (estimator_id.startswith("ease_")
                    and settings.get("ease_cv_target") != "final_ease_beta"):
                # Read only final-coefficient EASE results.
                continue
            label = ESTIMATOR_LABELS.get(estimator_id)
            if label in TABLE_ROWS[model]:
                rows.append({"model": model, "n": n, "d": int(rec.get("d", 0)),
                             "estimator": label, "ratio": float(rec["ratio"])})

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("no results for the main-paper tables")
    if data.duplicated(["model", "n", "d", "estimator"]).any():
        raise ValueError("duplicate paper rows: choose one run per method and setting")
    tables = {}
    for model, part in data.groupby("model"):
        if model in ("protein", "miami"):
            table = part.pivot(index="estimator", columns="n", values="ratio")
            tables[model, None] = check_table(table, model, SAMPLE_SIZES)
        else:
            for n, part_n in part.groupby("n"):
                table = part_n.pivot(index="estimator", columns="d", values="ratio")
                tables[model, n] = check_table(table, model, DIMENSIONS)
    return tables


def latex_table(blocks, caption, simulation=True):
    """Combine result tables and mark the largest RE in each column in bold."""
    columns = list(blocks[0][1].columns)
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + ("ll" if simulation else "l") + "r" * len(columns) + "}",
        r"\toprule",
        ("Model & " if simulation else "") + "Procedure & "
        + " & ".join(map(str, columns)) + r" \\",
    ]
    for label, table in blocks:
        lines.append(r"\midrule")
        best = table.max()
        for i, (estimator, row) in enumerate(table.iterrows()):
            cells = [rf"\textbf{{{v:.3f}}}" if v == best[c] else f"{v:.3f}" for c, v in row.items()]
            first = ""
            if simulation:
                first = f"({label}) & " if i == 0 else " & "
            lines.append(first + f"\\textsf{{{estimator}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\caption{" + caption + "}",
              r"\end{table}", ""]
    return "\n".join(lines)


def export(tables, output):
    models = ["M1", "M2", "M3", "M4"]
    simulation = [(model, tables[model, 500]) for model in models if (model, 500) in tables]
    if simulation and len(simulation) != len(models):
        raise ValueError("Table 1 needs completed M1, M2, M3 and M4 results")
    for (model, n), table in tables.items():
        real = model in ("protein", "miami")
        folder = output / ("realdata" if real else "simulation") / model
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder / f"{model}_table.csv", float_format="%.17g")
        if real:
            caption = f"{model.capitalize()}: RE by labeled sample size $n$, $N=5000$ and 100 replications."
        else:
            caption = f"{model}: RE by dimension $d$, $n={n}$, $N=5000$ and 100 replications."
        tex = latex_table([(model, table)], caption, simulation=not real)
        (folder / f"{model}_table.tex").write_text(tex, encoding="utf-8")
    if simulation:
        caption = "RE by dimension $d$, $n=500$, $N=5000$ and 100 replications."
        tex = latex_table(simulation, caption)
        (output / "simulation" / "simulation_table.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Make tables from saved results.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", action="store_true", help="use saved CSVs in results")
    source.add_argument("--runs", nargs="+", help="output folders or quoted glob patterns")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    if args["reference"]:
        result_tables = read_reference()
    else:
        folders = []
        for pattern in args["runs"]:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"no run folders match {pattern}")
            folders.extend(matches)
        result_tables = read_runs(folders)
    output = args["output"] or ROOT / "outputs"
    export(result_tables, output)
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()


