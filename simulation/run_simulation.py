"""Simulation study.

From the command line:

    python simulation/run_simulation.py s1 --estimators sta_sbf sta_csbf --n 500 --jobs 8

or from Python:

    config = SimulationConfig(model="s1", estimators=["sta_sbf", "pi"], n=200, reps=20)
    summary, records = run_simulation(config)
    ratio_table(summary)

Each run writes outputs/simulation/<run name>/ with settings.json, summary.csv
(RE by d and estimator) and records.pkl (one row per fit).
"""

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimation_methods.run_estimator import ESTIMATORS, Sample, fit_estimator
from estimation_methods.utils import kfold, ols_beta, sq_error
from experiment.runner import (
    new_output_dir, risk_ratio, run_replications, save_results, save_settings,
)
from simulation.models import MODELS, generate, projection_target

OUTPUT = ROOT / "outputs" / "simulation"


@dataclass
class SimulationConfig:
    model: str = "s1"
    estimators: tuple = ("sta_sbf", "sta_csbf", "lp_nw_n1_5", "lp_ll_n1_5",
                         "lp_nw_theory", "lp_ll_theory", "krr_m", "krr_r")
    dimensions: tuple = tuple(range(2, 16))
    n: int = 500
    N: int = 5000
    reps: int = 100
    seed: int = 1
    noise_sigma: float = 1.0
    K: int = 5  # EASE cross-fitting blocks
    r: int = 2  # EASE SIR directions
    H: int = None  # EASE SIR slices; n // 5 when None
    cv_folds: int = 10
    jobs: int = 1
    run_name: str = None


def run_one(task):
    """Generate one sample, fit each estimator and collect its result."""
    config, d, rep = task
    seed = config.seed + rep
    n, N = config.n, config.N
    X, y, X_u, m_u = generate(config.model, n, N, d, seed, config.noise_sigma)
    beta0 = projection_target(config.model, d)
    beta_ols = ols_beta(X, y)
    info = {
        "model": config.model,
        "d": d,
        "rep": rep,
        "seed": seed,
        "n": n,
        "N": N,
        "beta_hat_ols": beta_ols,
        "coefficient_squared_error_ols": sq_error(beta_ols, beta0),
    }
    folds = kfold(n, config.cv_folds, seed)
    sample = Sample(seed, X, y, X_u, beta0, folds, m_true=m_u, info=info)
    cache = {}
    records = []
    for name in config.estimators:
        record = fit_estimator(
            sample, name, K=config.K, r=config.r, H=config.H,
            cv_folds=config.cv_folds, cache=cache,
        )
        records.append(record)
    return records


def run_simulation(config):
    """Run every dimension and replication of one setting; returns (summary, records)."""
    config = replace(config, estimators=list(config.estimators),
                     dimensions=list(config.dimensions), H=config.H or config.n // 5)
    run_name = config.run_name or f"{config.model}_n{config.n}_N{config.N}"
    folder = new_output_dir(OUTPUT / run_name)
    save_settings(folder, {**asdict(config), "run_name": folder.name})

    tasks = list(itertools.product([config], config.dimensions, range(config.reps)))
    records = run_replications(tasks, run_one, config.jobs)
    records = records.sort_values(["d", "rep", "estimator_id"], ignore_index=True)
    summary = risk_ratio(records, ["model", "n", "N", "d", "estimator_id"])
    print(ratio_table(summary, digits=3).to_string())
    save_results(folder, summary[["d", "estimator_id", "ratio"]], records)
    print(f"saved to {folder}")
    return summary, records


def run_labeled_size_sweep(config, labeled_sizes, unlabeled_sizes=None):
    # One run per n, paired with unlabeled_sizes (or config.N for all of them).
    # With config.H = None each run uses its own n // 5.
    if unlabeled_sizes is None:
        unlabeled_sizes = config.N
    if np.isscalar(unlabeled_sizes):
        unlabeled_sizes = [unlabeled_sizes] * len(labeled_sizes)
    base = config.run_name or config.model
    results = [
        run_simulation(replace(config, n=n, N=N, run_name=f"{base}_n{n}_N{N}"))
        for n, N in zip(labeled_sizes, unlabeled_sizes, strict=True)
    ]
    return pd.concat([summary for summary, _ in results], ignore_index=True), results


def run_unlabeled_size_sweep(config, unlabeled_sizes):
    """run_simulation for each N with n fixed at config.n."""
    return run_labeled_size_sweep(config, [config.n] * len(unlabeled_sizes), unlabeled_sizes)


def ratio_table(summary, index="d", digits=4):
    table = summary.pivot_table(index=index, columns="estimator_id", values="ratio",
                                aggfunc="first")
    return table.round(digits)


def main():
    default = SimulationConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, default=default.estimators)
    parser.add_argument("--dimensions", nargs="+", type=int, default=default.dimensions)
    parser.add_argument("--n", type=int, default=default.n)
    parser.add_argument("--N", type=int, default=default.N)
    parser.add_argument("--reps", type=int, default=default.reps)
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--noise-sigma", type=float, default=default.noise_sigma)
    parser.add_argument("--K", type=int, default=default.K, help="EASE cross-fitting blocks")
    parser.add_argument("--r", type=int, default=default.r, help="EASE SIR directions")
    parser.add_argument("--H", type=int, help="EASE SIR slices (default n // 5)")
    parser.add_argument("--cv-folds", type=int, default=default.cv_folds)
    parser.add_argument("--jobs", type=int, default=default.jobs)
    parser.add_argument("--run-name")
    parser.add_argument("--print-config", action="store_true", help="print the settings and exit")
    args = vars(parser.parse_args())
    show_only = args.pop("print_config")

    config = SimulationConfig(**args)
    print(json.dumps({**asdict(config), "H": config.H or config.n // 5}, indent=2))
    if not show_only:
        run_simulation(config)


if __name__ == "__main__":
    main()
