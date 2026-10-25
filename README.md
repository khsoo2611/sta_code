## Overview

This repository contains the data, code, and results for our paper, "Structural target-aware semi-supervised linear prediction."

## Estimation Methods

`estimation_methods/` contains the following methods.

| File | Description |
|---|---|
| `ata_sbf.py` | ATA-SBF using smooth backfitting. |
| `ata_csbf.py` | ATA-CSBF using weighted and constrained smooth backfitting. |
| `nw.py` | Nadaraya-Watson regression. |
| `ll.py` | Local linear regression. |
| `krr_m.py` | Kernel ridge regression with the Matern kernel. |
| `krr_r.py` | Kernel ridge regression with the RBF kernel. |
| `ease.py` | EASE with SS-SIR dimension reduction. |
| `pi.py` | The partial-information estimator of Azriel et al. |

`utils.py` contains shared calculations, cross-validation, result saving, and table functions.

`simulation/models.py` contains M1-M4, sample generation, and the true coefficients. `simulation/run_simulation.py` runs the simulation experiments. `realdata/run_realdata.py` loads the real data and runs the repeated sampling experiments.

Ordinary NW and LL use the following bandwidth grids, defined in `utils.py`.

| Mode | Used for | Bandwidth candidates |
|---|---|---|
| `mode_1` | Simulation | 20 log-spaced values from `0.35*n^(-1/5)` to `0.5`. |
| `mode_2` | Real data (Protein and Miami) | `n^(-k/(21*d))`, with `k = 1, ..., 20`. |

## Data

### Protein

`realdata/data/protein/` contains the Protein dataset from the UCI Machine Learning Repository. The dataset has 45,730 observations. The response is RMSD, and the covariates are F1-F9.

`CASP.csv` is the original data. `CASP_processed.csv` contains F1-F9 and RMSD, the columns used in the experiments. The covariates are rescaled to [0,1] when the data are loaded.

### Miami

`realdata/data/miami/` contains MiamiHousing2016 from OpenML, data ID 43093. The dataset has 13,932 observations.

`miami_openml_43093.arff` is the original data, and `miami_openml_43093.csv` is its CSV version. `miami_openml_43093_processed.csv` contains the 12 covariates used in the paper and the response `LOG_SALE_PRC`, the natural logarithm of the sale price.

Each data folder contains `DATA_LICENSE.md` with the data source and usage terms.

The real-data notebooks make the processed CSV only when it is missing. If it already exists, they use that file.

## Results

`results/` contains the saved experiment results for M1-M4, Protein, and Miami. New runs do not replace these files.

RE, called `ratio` in the code, is the sum of the OLS squared coefficient errors divided by the sum for the method over the replications. A value above 1 means a smaller coefficient error than OLS.

### Simulation

`results/simulation/` contains the M1, M2, M3, and M4 folders.

Each model folder contains one result CSV: `M1.csv`, `M2.csv`, `M3.csv`, or `M4.csv`. All four use n = 500.

Rows are estimation methods, and columns are covariate dimensions. Each model folder contains `settings.json` with the experiment settings.

### Real Data

`results/realdata/miami/` contains `miami.csv`, and `results/realdata/protein/` contains `protein.csv`. Rows are estimation methods, and columns are labeled sample sizes. Each folder contains `settings.json`.

Both datasets use n = 100, 200, ..., 1000 and N = 5000.

### Example

Run `simulation/runner/M1_runner.ipynb` with the default n = 500 and N = 5000. It saves the following files:

```text
outputs/simulation/M1/M1_n500_N5000/
  summary.csv
  records.pkl
  settings.json
```

`outputs/` is created when an experiment starts. Other simulation runs use `outputs/simulation/<model>/<run_name>/`. Real-data runs use `outputs/realdata/<dataset>/<run_name>/`.

If the run folder already exists, `_2`, `_3`, and so on are added to its name.

| File | Contents |
|---|---|
| `summary.csv` | One row per method and dimension or sample size, with `estimator_id` and `ratio`. |
| `records.pkl` | Results from each replication, including coefficients, selected tuning values, and CV scores. |
| `settings.json` | Sample sizes, seed, replications, methods, and other run settings. |

The table export in `utils.py` reads saved results and writes CSV and LaTeX files under `outputs/` by default.

## Running Code

We use Python 3.13.5.

Open a terminal in the code folder and run:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

In JupyterLab, open the notebook for the experiment you want to run.

| Experiment | Notebook |
|---|---|
| M1 | `simulation/runner/M1_runner.ipynb` |
| M2 | `simulation/runner/M2_runner.ipynb` |
| M3 | `simulation/runner/M3_runner.ipynb` |
| M4 | `simulation/runner/M4_runner.ipynb` |
| Miami | `realdata/runner/miami_runner.ipynb` |
| Protein | `realdata/runner/protein_runner.ipynb` |

Run the cells in order, from top to bottom. The last cell uses the `summary` from the experiment cell. It shows RE to three decimal places, with the largest value in each column in bold. It does not save a LaTeX file.

The last cell does not read the saved CSVs. To view saved results after restarting the kernel, use the table commands below without rerunning the experiments.

The default settings use 100 replications and 12 workers. You can change the sample sizes, replications, seed, methods, and workers in the settings cell. Simulation notebooks also include the covariate dimensions.

`jobs` is the maximum number of worker processes running at the same time. Use a smaller value if needed for your computer.

The Miami and Protein notebooks include the preprocessing code. New results are saved under `outputs/`.

### Making Tables

Run these commands in the code folder.

To make tables from the saved CSVs in `results/`, without running experiments:

```bash
python utils.py --reference
```

To make tables from new simulation runs:

```bash
python utils.py --runs "outputs/simulation/M[1-4]/M*_n500_N5000"
```

This needs completed M1-M4 runs, each with n = 500 and d = 2, ..., 15.

For new Protein runs:

```bash
python utils.py --runs "outputs/realdata/protein/protein_n*_N5000"
```

For Miami, replace `protein` with `miami` in both parts of the path. Real-data export needs all sample sizes from n = 100 to 1000, in steps of 100.

The new-run commands expect the full notebook settings, including 100 replications, N = 5000, the saved seeds, and the CV settings. They check these settings before making the tables.

These examples use the default folder names for the first run. For later runs, use the actual folder names, including `_2`, `_3`, and so on. Choose only one run for each setting.

The exported files are:

| Table | Files |
|---|---|
| Each simulation model | `outputs/simulation/<model>/<model>_table.csv` and `<model>_table.tex` |
| M1-M4 combined | `outputs/simulation/simulation_table.tex` |
| Each real dataset | `outputs/realdata/<dataset>/<dataset>_table.csv` and `<dataset>_table.tex` |

Use `--output <folder>` to save the tables in another folder. The LaTeX files contain table blocks to copy into a document. They use the `booktabs` and `graphicx` packages.
