#!/usr/bin/env bash
# Rerun one experiment of the paper.
#
#   bash reproduce.sh M3            print the settings of every run
#   bash reproduce.sh M3 --run      run them (JOBS=12 worker processes by default)
#
# Tables and figures are then made with
#   python make_results.py --runs outputs/simulation/M3_n*_N5000
set -euo pipefail
cd "$(dirname "$0")"

EXPERIMENT=${1:-}
MODE=${2:---show}
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

usage() {
    echo "usage: bash reproduce.sh M1|M2|M3|M4|miami|protein [--show|--run]" >&2
    exit 2
}
[[ "$MODE" == --show || "$MODE" == --run ]] || usage

# EASE(F)-* in the tables is ease_f_*, EASE-* is ease_*.
EASE=(ease_f_nw ease_f_ll ease_f_m ease_f_r ease_nw ease_ll ease_m ease_r)

case "$EXPERIMENT" in
    M1)
        SCRIPT=simulation/run_simulation.py; TARGET=s1; SEED=1; SIZES=(500)
        ESTIMATORS=(sta_sbf sta_csbf lp_nw_n1_5 lp_ll_n1_5) ;;
    M2)
        SCRIPT=simulation/run_simulation.py; TARGET=s2; SEED=1; SIZES=(500)
        ESTIMATORS=(sta_sbf sta_csbf krr_m) ;;
    M3|M4)
        SCRIPT=simulation/run_simulation.py; SEED=1; SIZES=(100 200 300 400 500)
        TARGET=additive_quartic_c5
        [[ "$EXPERIMENT" == M4 ]] && TARGET=interaction1_all1
        ESTIMATORS=(sta_sbf sta_csbf lp_nw_n1_5 lp_ll_n1_5 krr_m krr_r "${EASE[@]}" pi) ;;
    miami|protein)
        SCRIPT=real_data/run_realdata.py; TARGET=$EXPERIMENT
        SIZES=(100 200 300 400 500 600 700 800 900 1000)
        SEED=202605310
        [[ "$EXPERIMENT" == protein ]] && SEED=202607196
        ESTIMATORS=(sta_sbf sta_csbf lp_nw_theory lp_ll_theory krr_m krr_r "${EASE[@]}" pi) ;;
    *)
        usage ;;
esac

for n in "${SIZES[@]}"; do
    ARGS=("$TARGET" --estimators "${ESTIMATORS[@]}"
          --n "$n" --N 5000 --reps 100 --seed "$SEED"
          --K 5 --r 2 --H $((n / 5)) --cv-folds 10
          --jobs "$JOBS" --run-name "${EXPERIMENT}_n${n}_N5000")
    if [[ "$SCRIPT" == simulation/* ]]; then
        ARGS+=(--dimensions {2..15} --noise-sigma 1)
    fi
    if [[ "$MODE" == --show ]]; then
        ARGS+=(--print-config)
    fi
    "$PYTHON" "$SCRIPT" "${ARGS[@]}"
done
