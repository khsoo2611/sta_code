"""Model M3 in Table 1. Run: python simulation/M3.py"""

import os
import sys
from pathlib import Path

# Each worker uses one BLAS thread.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.run_simulation import SimulationConfig, run_simulation

# Experiment settings
n = 500
N = 5000
reps = 100
dimensions = range(2, 16)
seed = 1
noise_sigma = 1.0
jobs = 12
estimators = [
    "ata_sbf", "ata_csbf", "lp_nw_n1_5", "lp_ll_n1_5", "krr_m", "krr_r",
    "ease_nw", "ease_ll", "ease_m", "ease_r", "pi",
]

# EASE: cross-fitting folds, reduced dimension, SIR slices, and inner CV folds
K = 5
r = 2
H = n // 5
cv_folds = 10  # also used by the direct estimators

experiment = SimulationConfig(
    model="M3",
    estimators=estimators,
    dimensions=dimensions,
    n=n,
    N=N,
    reps=reps,
    seed=seed,
    noise_sigma=noise_sigma,
    K=K,
    r=r,
    H=H,
    cv_folds=cv_folds,
    jobs=jobs,
    run_name=f"M3_n{n}_N{N}",
)

if __name__ == "__main__":
    summary, records = run_simulation(experiment)
