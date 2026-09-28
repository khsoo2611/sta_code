"""Table 1 (M1-M4) and Table 2 (Protein) of the main paper.

    python make_results.py --reference
    python make_results.py --runs "outputs/simulation/M[1-4]_n500_N5000"

Use --reference to format the archived ratios, or --runs to use finished runs.
Neither option starts an experiment.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DIMENSIONS = list(range(2, 16))
SAMPLE_SIZES = list(range(100, 1001, 100))
METHODS = [
    "ATA-SBF", "ATA-CSBF", "LP-NW", "LP-LL", "KRR-M", "KRR-R",
    "EASE-NW", "EASE-LL", "EASE-M", "EASE-R", "PI",
]
TABLE_ROWS = {
    "M1": ["LP-LL", "ATA-SBF", "ATA-CSBF"],
    "M2": ["KRR-M", "ATA-SBF", "ATA-CSBF"],
    "M3": METHODS, "M4": METHODS,
    "protein": METHODS,
}
MODEL_LABELS = {
    "s1": "M1", "s2": "M2", "additive_quartic_c5": "M3", "interaction1_all1": "M4",
    "M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4", "protein": "protein",
}
ESTIMATOR_LABELS = {
    "ata_sbf": "ATA-SBF", "ata_csbf": "ATA-CSBF", "lp_nw": "LP-NW", "lp_ll": "LP-LL",
    "krr_m": "KRR-M", "krr_r": "KRR-R", "pi": "PI",
    "ease_nw": "EASE-NW", "ease_ll": "EASE-LL", "ease_m": "EASE-M", "ease_r": "EASE-R",
}
# estimator ids of runs made before the estimators were renamed
OLD_IDS = {
    "sta_sbf": "ata_sbf", "sta_csbf": "ata_csbf", "sbf": "ata_sbf", "c_sbf": "ata_csbf",
    "nw_n1_5_imp": "lp_nw_n1_5", "nw_theory_imp": "lp_nw_theory",
    "local_linear_n1_5_imp": "lp_ll_n1_5", "local_linear_theory_imp": "lp_ll_theory",
    "matern_krr_imp": "krr_m", "rbf_krr_imp": "krr_r", "pi_imp": "pi",
    "ease_nw_low": "ease_nw", "ease_ll_low": "ease_ll",
    "ease_matern_krr_low": "ease_m", "ease_rbf_krr_low": "ease_r",
}
# row names in the archived CSVs
OLD_LABELS = {
    "STA-SBF": "ATA-SBF", "STA-CSBF": "ATA-CSBF",
    "EASE(L)-NW": "EASE-NW", "EASE(L)-LL": "EASE-LL",
    "EASE(L)-Matern": "EASE-M", "EASE(L)-RBF": "EASE-R",
}


def check_table(table, model, columns):
    rows = TABLE_ROWS[model]
    complete = set(rows).issubset(table.index) and set(table.columns) == set(columns)
    if not complete or not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"{model}: expected rows {rows} and columns {columns}")
    table = table.loc[rows, columns].astype(float)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"{model}: missing or non-finite RE values")
    table.index.name = "estimator"
    return table


def paper_label(estimator_id, real):
    """Row name in the paper, or None for estimators the paper does not show."""
    estimator_id = OLD_IDS.get(estimator_id, estimator_id)
    if estimator_id.startswith("lp_"):
        # LP uses the n^(-1/5) grid in the simulations and the theory grid on real data
        grid = "_theory" if real else "_n1_5"
        if not estimator_id.endswith(grid):
            return None
        estimator_id = estimator_id.removesuffix(grid)
    return ESTIMATOR_LABELS.get(estimator_id)


def read_reference():
    folder = ROOT / "results" / "reference" / "csv"
    tables = {}
    for model in TABLE_ROWS:
        if model in ("M3", "M4"):
            files = {500: f"{model}_n500.csv"}
        elif model == "protein":
            files = {None: f"{model}.csv"}
        else:
            files = {500: f"{model}.csv"}
        for n, name in files.items():
            table = pd.read_csv(folder / model / name, index_col="estimator",
                                float_precision="round_trip")
            table = table.rename(index=OLD_LABELS)
            table.columns = table.columns.astype(int)
            tables[model, n] = check_table(table, model, SAMPLE_SIZES if n is None else DIMENSIONS)
    return tables


def read_runs(folders):
    rows = []
    for folder in map(Path, folders):
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        model = MODEL_LABELS[settings.get("model") or settings["dataset"]]
        n = settings["n"]
        real = model == "protein"
        if (real and n not in SAMPLE_SIZES) or (not real and n != 500):
            raise ValueError(f"{folder}: n={n} is not in the main-paper table")
        paper = {"N": 5000, "reps": 100, "cv_folds": 10,
                 "seed": 202607196 if real else 1}
        if model in ("M3", "M4", "protein"):
            paper.update(K=5, r=2, H=n // 5)
        if not real:
            paper["noise_sigma"] = 1.0
        wrong = {key: settings.get(key) for key in paper if settings.get(key) != paper[key]}
        if wrong:
            raise ValueError(f"{folder}: settings differ from the paper: {wrong}")

        summary = pd.read_csv(folder / "summary.csv", float_precision="round_trip")
        for rec in summary.to_dict(orient="records"):
            label = paper_label(rec["estimator_id"], real)
            if label in TABLE_ROWS[model]:
                rows.append({"model": model, "n": n, "d": int(rec.get("d", 0)),
                             "estimator": label, "ratio": float(rec["ratio"])})

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("no results for the main-paper tables")
    if data.duplicated(["model", "n", "d", "estimator"]).any():
        raise ValueError("the same result appears in more than one run")
    tables = {}
    for model, part in data.groupby("model"):
        if model == "protein":
            table = part.pivot(index="estimator", columns="n", values="ratio")
            tables[model, None] = check_table(table, model, SAMPLE_SIZES)
        else:
            for n, part_n in part.groupby("n"):
                table = part_n.pivot(index="estimator", columns="d", values="ratio")
                tables[model, n] = check_table(table, model, DIMENSIONS)
    return tables


def latex_table(blocks, caption, simulation=True):
    """Stack (label, table) blocks in one tabular; the largest RE of each column is bold."""
    columns = list(blocks[0][1].columns)
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + ("ll" if simulation else "l") + "r" * len(columns) + "}",
        r"\toprule",
        ("Model & " if simulation else "") + "Procedure & "
        + " & ".join(map(str, columns)) + r" \\",
    ]
    for label, table in blocks:
        lines.append(r"\midrule")
        best = table.max()
        for i, (estimator, row) in enumerate(table.iterrows()):
            cells = [rf"\textbf{{{v:.3f}}}" if v == best[c] else f"{v:.3f}" for c, v in row.items()]
            # Table 1 uses LL in M1 and NW/LL in M3, without the LP prefix.
            if label in ("M1", "M3"):
                estimator = estimator.removeprefix("LP-")
            first = ""
            if simulation:
                first = f"({label}) & " if i == 0 else " & "
            lines.append(first + f"\\textsf{{{estimator}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\caption{" + caption + "}",
              r"\end{table}", ""]
    return "\n".join(lines)


def plot_simulation(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ATA-SBF": ("#126782", "o"), "ATA-CSBF": ("#b1253b", "s"),
        "LP-NW": ("#238b45", "^"), "LP-LL": ("#df8a22", "D"),
        "KRR-M": ("#6350a3", "D"),
    }
    with plt.rc_context({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"}):
        fig, ax = plt.subplots(figsize=(8.4, 5.1))
        for method, row in table.iterrows():
            color, marker = styles[method]
            ax.plot(table.columns, row, label=method, color=color, marker=marker,
                    markerfacecolor=color, markersize=5, linewidth=1.5)
        ax.set(xlabel=r"$d$", ylabel="RE", xlim=(1.6, 15.4))
        ax.set_ylim(0, float(table.to_numpy().max()) * 1.10)
        ax.set_xticks(DIMENSIONS)
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=len(table), frameon=False)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.23)
        fig.savefig(path)
        plt.close(fig)


def export(tables, output):
    models = ["M1", "M2", "M3", "M4"]
    simulation = [(model, tables[model, 500]) for model in models if (model, 500) in tables]
    if simulation and len(simulation) != len(models):
        raise ValueError("Table 1 needs completed M1, M2, M3 and M4 results")
    for sub in ("csv", "latex", "figures"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    for (model, n), table in tables.items():
        table.to_csv(output / "csv" / f"{model}.csv", float_format="%.17g")
        if model in ("M1", "M2"):
            plot_simulation(table, output / "figures" / f"{model}.pdf")
    if simulation:
        caption = "RE by dimension $d$, $n=500$, $N=5000$ and 100 replications."
        tex = latex_table(simulation, caption)
        (output / "latex" / "Table1.tex").write_text(tex, encoding="utf-8")
    if ("protein", None) in tables:
        caption = "Protein: RE by labeled sample size $n$, $N=5000$ and 100 replications."
        tex = latex_table([("Protein", tables["protein", None])], caption, simulation=False)
        (output / "latex" / "Table2.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", action="store_true", help="use results/reference/csv")
    source.add_argument("--runs", nargs="+", help="output folders or quoted glob patterns")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.reference:
        tables = read_reference()
    else:
        folders = []
        for pattern in args.runs:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"no run folders match {pattern}")
            folders.extend(matches)
        tables = read_runs(folders)
    default = "reference_tables" if args.reference else "recomputed_tables"
    output = args.output or ROOT / "outputs" / default
    export(tables, output)
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
