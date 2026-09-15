import concurrent.futures as cf
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
    # Fields missing from every record are omitted from records.pkl.
    rows = []
    for record in records:
        row = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


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
