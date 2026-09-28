"""Model M1 in Table 1. Run: python simulation/M1.py"""

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
cv_folds = 10
jobs = 12
estimators = ["ata_sbf", "ata_csbf", "lp_ll_n1_5"]

experiment = SimulationConfig(
    model="M1",
    estimators=estimators,
    dimensions=dimensions,
    n=n,
    N=N,
    reps=reps,
    seed=seed,
    noise_sigma=noise_sigma,
    cv_folds=cv_folds,
    jobs=jobs,
    run_name=f"M1_n{n}_N{N}",
)

if __name__ == "__main__":
    summary, records = run_simulation(experiment)
