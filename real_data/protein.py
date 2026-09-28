"""Protein data in Table 2. Run: python real_data/protein.py"""

import os
import sys
from pathlib import Path

# Each worker uses one BLAS thread.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_data.run_realdata import run_realdata

# Experiment settings
labeled_sizes = range(100, 1001, 100)
N = 5000
reps = 100
seed = 202607196
jobs = 12
estimators = [
    "ata_sbf", "ata_csbf", "lp_nw_theory", "lp_ll_theory", "krr_m", "krr_r",
    "ease_nw", "ease_ll", "ease_m", "ease_r", "pi",
]

# EASE: cross-fitting folds, reduced dimension, and inner CV folds
K = 5
r = 2
cv_folds = 10  # also used by the direct estimators

if __name__ == "__main__":
    for n in labeled_sizes:
        H = n // 5  # SIR slices
        summary, records = run_realdata(
            "protein",
            estimators=estimators,
            n=n,
            N=N,
            reps=reps,
            seed=seed,
            K=K,
            r=r,
            H=H,
            cv_folds=cv_folds,
            jobs=jobs,
            run_name=f"protein_n{n}_N{N}",
        )
