"""Run simulation experiments for M1-M4.

Settings
--------
model: Model name.
estimators: Methods to compare.
            NW and LL share one bandwidth grid, capped at 0.8.
            EASE uses final EASE coefficient CV.
dimensions: Numbers of covariates to examine.
n: Number of labeled observations, with covariates and responses.
N: Number of unlabeled observations, with covariates only.
reps: Number of independent repetitions at each dimension.
seed: Random seed for the first repetition.
noise_sigma: Standard deviation of the normal response error.
K: Number of EASE cross-fitting groups. Each fit uses K - 1 groups
   to predict the remaining group.
r: Number of covariates after SS-SIR dimension reduction in EASE.
H: Number of response intervals used by SS-SIR. If None, use n // 5.
cv_folds: Number of groups for tuning CV on the labeled sample.
          EASE scores predictions from its final coefficients.
jobs: Number of worker processes for running repetitions in parallel.
run_name: Output folder name. If None, use the model and sample sizes.

Functions
---------
SimulationConfig stores these settings.
run_one_simulation generates data and fits the methods for one repetition.
run_simulation runs all dimensions and saves the results.
It returns summary (RE values) and records (results from each repetition).
"""

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.models import MODELS, generate, projection_target
from utils import (
    ESTIMATORS, LP_BANDWIDTH_MAX, Sample, fit_estimator, kfold, new_output_dir, ols_beta,
    ratio_table, risk_ratio, run_replications, save_results, save_settings, sq_error,
)

SIMULATION_OUTPUT = ROOT / "outputs" / "simulation"


DEFAULT_ESTIMATORS = {
    "M1": ("lp_ll", "ata_sbf", "ata_csbf"),
    "M2": ("krr_m", "ata_sbf", "ata_csbf"),
    "M3": (
        "ata_sbf", "ata_csbf", "lp_nw", "lp_ll", "krr_m", "krr_r",
        "ease_nw", "ease_ll", "ease_m", "ease_r", "pi",
    ),
    "M4": (
        "ata_sbf", "ata_csbf", "lp_nw", "lp_ll", "krr_m", "krr_r",
        "ease_nw", "ease_ll", "ease_m", "ease_r", "pi",
    ),
}


@dataclass
class SimulationConfig:
    model: str = "M1"
    estimators: tuple = None
    dimensions: tuple = tuple(range(2, 16))
    n: int = 500
    N: int = 5000
    reps: int = 100
    seed: int = 1
    noise_sigma: float = 1.0
    K: int = 5  # Groups for EASE cross-fitting
    r: int = 2  # Dimension after SS-SIR
    H: int = None  # SS-SIR response intervals. If None, use n // 5.
    cv_folds: int = 10
    jobs: int = 1
    run_name: str = None


    def __post_init__(self):
        if self.estimators is None:
            self.estimators = DEFAULT_ESTIMATORS[self.model]


def run_one_simulation(task):
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
    records = []
    for name in config.estimators:
        record = fit_estimator(
            sample, name, K=config.K, r=config.r, H=config.H,
            cv_folds=config.cv_folds,
        )
        records.append(record)
    return records


def run_simulation(config):
    """Run all dimensions and return the RE summary and repetition records."""
    config = replace(config, estimators=list(config.estimators),
                     dimensions=list(config.dimensions), H=config.H or config.n // 5)
    run_name = config.run_name or f"{config.model}_n{config.n}_N{config.N}"
    folder = new_output_dir(SIMULATION_OUTPUT / config.model / run_name)
    settings = {**asdict(config), "run_name": folder.name,
                "sbf_normalization": "exact", "csbf_normalization": "exact"}
    if "ata_csbf" in config.estimators:
        settings["csbf_weight_source"] = "unlabeled"
        settings["csbf_weight_sample_size"] = config.N
    if any(name in ("lp_nw", "lp_ll") for name in config.estimators):
        settings["lp_bandwidth_max"] = LP_BANDWIDTH_MAX
    if any(name.startswith("ease_") for name in config.estimators):
        settings["ease_cv_target"] = "final_ease_beta"
    save_settings(folder, settings)

    tasks = list(itertools.product([config], config.dimensions, range(config.reps)))
    records = run_replications(tasks, run_one_simulation, config.jobs)
    records = records.sort_values(["d", "rep", "estimator_id"], ignore_index=True)
    summary = risk_ratio(records, ["model", "n", "N", "d", "estimator_id"])
    print(ratio_table(summary, config.model, digits=3).to_string())
    save_results(folder, summary[["d", "estimator_id", "ratio"]], records)
    print(f"saved to {folder}")
    return summary, records


def main():
    default = SimulationConfig()
    parser = argparse.ArgumentParser(description="Run M1, M2, M3 or M4.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS)
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
    settings = {**asdict(config), "H": config.H or config.n // 5,
                "sbf_normalization": "exact", "csbf_normalization": "exact"}
    if any(name in ("lp_nw", "lp_ll") for name in config.estimators):
        settings["lp_bandwidth_max"] = LP_BANDWIDTH_MAX
    if any(name.startswith("ease_") for name in config.estimators):
        settings["ease_cv_target"] = "final_ease_beta"
    print(json.dumps(settings, indent=2))
    if not show_only:
        run_simulation(config)


if __name__ == "__main__":
    main()
