"""Run repeated data splits for Miami or Protein.

Settings
--------
dataset: Data name, miami or protein.
estimators: Methods to compare.
n: Number of labeled observations, with covariates and responses.
N: Number of unlabeled observations, using covariates only.
reps: Number of random data splits.
seed: Random seed for the first split. If None, use the saved dataset seed.
K: Number of EASE cross-fitting groups. Each fit uses K - 1 groups
   to predict the remaining group.
r: Number of covariates after SS-SIR dimension reduction in EASE.
H: Number of response intervals used by SS-SIR. If None, use n // 5.
cv_folds: Number of groups for tuning CV. Direct methods use the labeled
          sample. EASE uses the K - 1 training groups.
jobs: Number of worker processes for running repetitions in parallel.
run_name: Output folder name. If None, use the dataset and sample sizes.

Functions
---------
load_dataset reads the processed CSV, removes rows with missing values,
scales each covariate to [0, 1], and computes the full-data OLS reference.
make_config stores the run settings.
run_one_realdata draws one split and fits the selected methods.
run_realdata repeats the splits and saves the results.
It returns summary (RE values) and records (results from each repetition).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    ESTIMATORS, Sample, fit_estimator, kfold, new_output_dir, ols_beta,
    ratio_table, risk_ratio, run_replications, save_results, save_settings, sq_error,
)

REALDATA_OUTPUT = ROOT / "outputs" / "realdata"


DATASETS = {
    "miami": {
        "csv": ROOT / "realdata" / "data" / "miami" / "miami_openml_43093_processed.csv",
        "target": "LOG_SALE_PRC",
        "seed": 202605310,
    },
    "protein": {
        "csv": ROOT / "realdata" / "data" / "protein" / "CASP_processed.csv",
        "target": "RMSD",
        "seed": 202607196,
    },
}


def load_dataset(name):
    spec = DATASETS[name]
    data = pd.read_csv(spec["csv"]).dropna()
    X = data.drop(columns=spec["target"]).to_numpy(dtype=float)
    y = data[spec["target"]].to_numpy(dtype=float)
    low = X.min(axis=0)
    span = X.max(axis=0) - low
    X = (X - low) / np.where(np.abs(span) <= 1e-12, 1.0, span)
    return {"name": name, "X": X, "y": y, "beta0": ols_beta(X, y)}


_data = None  # set in each worker process


def _set_data(data):
    global _data
    _data = data


def run_one_realdata(task):
    """Draw one labeled/unlabeled split, fit each estimator and collect its result."""
    config, rep = task
    data = _data
    seed = config["seed"] + rep
    n, N = config["n"], config["N"]
    rows = np.random.default_rng(seed).choice(len(data["y"]), size=n + N, replace=False)
    labeled = np.random.default_rng(seed + 1).choice(n + N, size=n, replace=False)
    unlabeled = np.delete(np.arange(n + N), labeled)
    X, y = data["X"][rows], data["y"][rows]
    beta0 = data["beta0"]
    beta_ols = ols_beta(X[labeled], y[labeled])
    info = {
        "dataset": data["name"],
        "d": X.shape[1],
        "rep": rep,
        "seed": seed,
        "n": n,
        "N": N,
        "beta_hat_ols": beta_ols,
        "coefficient_squared_error_ols": sq_error(beta_ols, beta0),
    }
    folds = kfold(n, config["cv_folds"], seed)
    sample = Sample(seed, X[labeled], y[labeled], X[unlabeled], beta0, folds, info=info)
    cache = {}
    records = []
    for name in config["estimators"]:
        record = fit_estimator(
            sample, name, K=config["K"], r=config["r"], H=config["H"],
            cv_folds=config["cv_folds"], cache=cache,
        )
        records.append(record)
    return records


def make_config(dataset, estimators, n, N, reps, seed, K, r, H, cv_folds, jobs):
    return {"dataset": dataset, "estimators": list(estimators), "n": n, "N": N, "reps": reps,
            "seed": DATASETS[dataset]["seed"] if seed is None else seed,
            "K": K, "r": r, "H": H or n // 5, "cv_folds": cv_folds, "jobs": jobs,
            "sbf_normalization": "exact", "csbf_normalization": "exact"}


def run_realdata(dataset, *, estimators, n, N, reps=100, seed=None, K=5, r=2, H=None,
                 cv_folds=10, jobs=1, run_name=None):
    """Repeat data splits and return the RE summary and repetition records."""
    config = make_config(
        dataset=dataset, estimators=estimators, n=n, N=N, reps=reps, seed=seed,
        K=K, r=r, H=H, cv_folds=cv_folds, jobs=jobs,
    )
    folder = new_output_dir(REALDATA_OUTPUT / dataset / (run_name or f"{dataset}_n{n}_N{N}"))
    save_settings(folder, {**config, "run_name": folder.name})
    data = load_dataset(dataset)
    tasks = [(config, rep) for rep in range(config["reps"])]
    records = run_replications(tasks, run_one_realdata, config["jobs"],
                               initializer=_set_data, initargs=(data,))
    records = records.sort_values(["rep", "estimator_id"], ignore_index=True)
    summary = risk_ratio(records, ["dataset", "n", "N", "estimator_id"])
    print(ratio_table(summary, config["dataset"], digits=3).to_string())
    save_results(folder, summary[["n", "estimator_id", "ratio"]], records)
    print(f"saved to {folder}")
    return summary, records


def main():
    parser = argparse.ArgumentParser(description="Run a real-data experiment.")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, help="default: the seed stored in DATASETS")
    parser.add_argument("--K", type=int, default=5, help="EASE cross-fitting blocks")
    parser.add_argument("--r", type=int, default=2, help="EASE SIR directions")
    parser.add_argument("--H", type=int, help="EASE SIR slices (default n // 5)")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-name")
    parser.add_argument("--print-config", action="store_true", help="print the settings and exit")

    args = vars(parser.parse_args())
    show_only = args.pop("print_config")
    run_name = args.pop("run_name")
    print(json.dumps(make_config(**args), indent=2))
    if not show_only:
        run_realdata(**args, run_name=run_name)


if __name__ == "__main__":
    main()
