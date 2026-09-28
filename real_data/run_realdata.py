"""Repeated random labeled/unlabeled splits. Table 2 settings are in protein.py.

From the command line:

    python real_data/run_realdata.py miami --estimators ata_sbf pi --n 500 --jobs 12

or from Python, run_realdata(...) for one (n, N) and run_sample_size_grid(...)
for every combination of several n and N.

Features are scaled to [0, 1] with the full-data minimum and maximum, and the
target beta0 is the OLS coefficient on the full data set. Results are written
to outputs/real_data/<run name>/.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from estimation_methods.run_estimator import ESTIMATORS, Sample, fit_estimator
from estimation_methods.utils import kfold, ols_beta, sq_error
from experiment.runner import (
    new_output_dir, risk_ratio, run_replications, save_results, save_settings,
)

OUTPUT = ROOT / "outputs" / "real_data"

# The processed CSVs are made by process_data.ipynb in each data folder.
DATASETS = {
    "miami": {
        "csv": HERE / "miami" / "data" / "miami_openml_43093_processed.csv",
        "target": "LOG_SALE_PRC",
        "features": ["LATITUDE", "LONGITUDE", "LND_SQFOOT", "TOT_LVG_AREA", "SPEC_FEAT_VAL",
                     "RAIL_DIST", "OCEAN_DIST", "WATER_DIST", "CNTR_DIST", "SUBCNTR_DI",
                     "HWY_DIST", "age"],
        "seed": 202605310,
    },
    "protein": {
        "csv": HERE / "protein" / "data" / "CASP_processed.csv",
        "target": "RMSD",
        "features": [f"F{i}" for i in range(1, 10)],
        "seed": 202607196,
    },
}


def load_dataset(name):
    spec = DATASETS[name]
    data = pd.read_csv(spec["csv"])[spec["features"] + [spec["target"]]].dropna()
    X = data[spec["features"]].to_numpy(dtype=float)
    y = data[spec["target"]].to_numpy(dtype=float)
    low = X.min(axis=0)
    span = X.max(axis=0) - low
    X = (X - low) / np.where(np.abs(span) <= 1e-12, 1.0, span)
    return {"name": name, "X": X, "y": y, "beta0": ols_beta(X, y)}


_data = None  # set in each worker process


def _set_data(data):
    global _data
    _data = data


def run_one(task):
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
            "K": K, "r": r, "H": H or n // 5, "cv_folds": cv_folds, "jobs": jobs}


def replicate(data, config):
    """All replications of one (n, N); returns (summary, records)."""
    tasks = [(config, rep) for rep in range(config["reps"])]
    records = run_replications(tasks, run_one, config["jobs"],
                               initializer=_set_data, initargs=(data,))
    records = records.sort_values(["rep", "estimator_id"], ignore_index=True)
    summary = risk_ratio(records, ["dataset", "n", "N", "estimator_id"])
    print(ratio_table(summary, digits=3).to_string(index=False))
    return summary, records


def run_realdata(dataset, *, estimators, n, N, reps=100, seed=None, K=5, r=2, H=None,
                 cv_folds=10, jobs=1, run_name=None):
    """Repeated splits for one (n, N); returns (summary, records)."""
    config = make_config(dataset, estimators, n, N, reps, seed, K, r, H, cv_folds, jobs)
    folder = new_output_dir(OUTPUT / (run_name or f"{dataset}_n{n}_N{N}"))
    save_settings(folder, {**config, "run_name": folder.name})
    summary, records = replicate(load_dataset(dataset), config)
    save_results(folder, summary[["n", "estimator_id", "ratio"]], records)
    print(f"saved to {folder}")
    return summary, records


def run_sample_size_grid(dataset, *, estimators, n_values, N_values, reps=100, seed=None,
                         K=5, r=2, H=None, cv_folds=10, jobs=1, run_name=None):
    """Repeated splits for every (n, N) in n_values x N_values, saved in one folder.

    With H = None every n uses its own n // 5.
    """
    folder = new_output_dir(OUTPUT / (run_name or f"{dataset}_size_grid"))
    save_settings(folder, {
        "dataset": dataset, "estimators": list(estimators),
        "n_values": list(n_values), "N_values": list(N_values), "reps": reps,
        "seed": DATASETS[dataset]["seed"] if seed is None else seed,
        "K": K, "r": r, "H": H, "cv_folds": cv_folds, "jobs": jobs, "run_name": folder.name,
    })
    data = load_dataset(dataset)
    summaries, records = [], []
    for n, N in itertools.product(n_values, N_values):
        config = make_config(dataset, estimators, n, N, reps, seed, K, r, H, cv_folds, jobs)
        summary, rec = replicate(data, config)
        summaries.append(summary)
        records.append(rec)
    summary = pd.concat(summaries, ignore_index=True)
    records = pd.concat(records, ignore_index=True)
    save_results(folder, summary[["n", "N", "estimator_id", "ratio"]], records)
    print(f"saved to {folder}")
    return summary, records


def ratio_table(summary, digits=4):
    table = summary[["n", "N", "estimator_id", "ratio"]]
    return table.sort_values(["n", "N", "estimator_id"], ignore_index=True).round({"ratio": digits})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
