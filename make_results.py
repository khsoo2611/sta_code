"""Tables and figures of the paper.

    python make_results.py --reference                           archived ratios
    python make_results.py --runs outputs/simulation/M3_n*_N5000  finished runs

The archived ratios are in results/reference/csv. Output goes to csv/, latex/
and figures/ under outputs/ (or --output).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DIMENSIONS = list(range(2, 16))
SAMPLE_SIZES = list(range(100, 1001, 100))
REAL = ("miami", "protein")
METHODS = [
    "STA-SBF", "STA-CSBF", "LP-NW", "LP-LL", "KRR-M", "KRR-R",
    "EASE(F)-NW", "EASE(F)-LL", "EASE(F)-M", "EASE(F)-R",
    "EASE-NW", "EASE-LL", "EASE-M", "EASE-R", "PI",
]
TABLE_ROWS = {
    "M1": METHODS[:4], "M2": ["STA-SBF", "STA-CSBF", "KRR-M"],
    "M3": METHODS, "M4": METHODS,
    "miami": ["OLS", *METHODS], "protein": ["OLS", *METHODS],
}
MODEL_LABELS = {
    "s1": "M1", "s2": "M2", "additive_quartic_c5": "M3", "interaction1_all1": "M4",
    "miami": "miami", "protein": "protein",
}
ESTIMATOR_LABELS = {
    "sta_sbf": "STA-SBF", "sta_csbf": "STA-CSBF", "lp_nw": "LP-NW", "lp_ll": "LP-LL",
    "krr_m": "KRR-M", "krr_r": "KRR-R", "pi": "PI",
    "ease_f_nw": "EASE(F)-NW", "ease_f_ll": "EASE(F)-LL",
    "ease_f_m": "EASE(F)-M", "ease_f_r": "EASE(F)-R",
    "ease_nw": "EASE-NW", "ease_ll": "EASE-LL", "ease_m": "EASE-M", "ease_r": "EASE-R",
}
# estimator ids of runs made before the estimators were renamed
OLD_IDS = {
    "sbf": "sta_sbf", "c_sbf": "sta_csbf",
    "nw_n1_5_imp": "lp_nw_n1_5", "nw_theory_imp": "lp_nw_theory",
    "local_linear_n1_5_imp": "lp_ll_n1_5", "local_linear_theory_imp": "lp_ll_theory",
    "matern_krr_imp": "krr_m", "rbf_krr_imp": "krr_r", "pi_imp": "pi",
    "ease_nw_low": "ease_nw", "ease_ll_low": "ease_ll",
    "ease_matern_krr_low": "ease_m", "ease_rbf_krr_low": "ease_r",
    "snp_nw_full": "ease_f_nw", "snp_ll_full": "ease_f_ll",
    "snp_matern_krr_full": "ease_f_m", "snp_rbf_krr_full": "ease_f_r",
}
# row names in the archived CSVs
OLD_LABELS = {
    "EASE(F)-Matern": "EASE(F)-M", "EASE(F)-RBF": "EASE(F)-R",
    "EASE(L)-NW": "EASE-NW", "EASE(L)-LL": "EASE-LL",
    "EASE(L)-Matern": "EASE-M", "EASE(L)-RBF": "EASE-R",
}


def check_table(table, model, columns):
    rows = TABLE_ROWS[model]
    complete = set(table.index) == set(rows) and set(table.columns) == set(columns)
    if not complete or table.isna().any().any():
        raise ValueError(f"{model}: expected rows {rows} and columns {columns}")
    table = table.loc[rows, columns].astype(float)
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
            files = {n: f"{model}_n{n}.csv" for n in range(100, 501, 100)}
        elif model in REAL:
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
        paper = {"N": 5000, "reps": 100, "K": 5, "r": 2, "H": n // 5, "cv_folds": 10}
        wrong = {key: settings.get(key) for key in paper if settings.get(key) != paper[key]}
        if wrong:
            raise ValueError(f"{folder}: settings differ from the paper: {wrong}")

        real = model in REAL
        summary = pd.read_csv(folder / "summary.csv", float_precision="round_trip")
        for rec in summary.to_dict(orient="records"):
            label = paper_label(rec["estimator_id"], real)
            if label is not None:
                rows.append({"model": model, "n": n, "d": int(rec.get("d", 0)),
                             "estimator": label, "ratio": float(rec["ratio"])})
        if real:
            rows.append({"model": model, "n": n, "d": 0, "estimator": "OLS", "ratio": 1.0})

    data = pd.DataFrame(rows)
    if data.duplicated(["model", "n", "d", "estimator"]).any():
        raise ValueError("the same result appears in more than one run")
    tables = {}
    for model, part in data.groupby("model"):
        if model in REAL:
            table = part.pivot(index="estimator", columns="n", values="ratio")
            tables[model, None] = check_table(table, model, SAMPLE_SIZES)
        else:
            for n, part_n in part.groupby("n"):
                table = part_n.pivot(index="estimator", columns="d", values="ratio")
                tables[model, n] = check_table(table, model, DIMENSIONS)
    return tables


def latex_table(blocks, caption):
    """Stack (label, table) blocks in one tabular; the largest RE of each column is bold."""
    columns = list(blocks[0][1].columns)
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(columns) + "}",
        r"\toprule",
        "Model & EstPrc & " + " & ".join(map(str, columns)) + r" \\",
    ]
    for label, table in blocks:
        lines.append(r"\midrule")
        best = table.max()
        for i, (estimator, row) in enumerate(table.iterrows()):
            cells = [rf"\textbf{{{v:.3f}}}" if v == best[c] else f"{v:.3f}" for c, v in row.items()]
            first = label if i == 0 else ""
            lines.append(f"{first} & \\textsf{{{estimator}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\caption{" + caption + "}",
              r"\end{table}", ""]
    return "\n".join(lines)


def plot_simulation(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "STA-SBF": ("#126782", "o"), "STA-CSBF": ("#b1253b", "s"),
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
    for sub in ("csv", "latex", "figures"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    for (model, n), table in tables.items():
        name = f"{model}_n{n}" if model in ("M3", "M4") else model
        table.to_csv(output / "csv" / f"{name}.csv", float_format="%.17g")
        if model in ("M1", "M2"):
            plot_simulation(table, output / "figures" / f"{model}.pdf")
        if model in REAL:
            caption = f"{model.title()}: RE by labeled sample size $n$ ($N=5000$)."
            tex = latex_table([(model.title(), table)], caption)
            (output / "latex" / f"{model}.tex").write_text(tex, encoding="utf-8")

    # the simulation tables show two models together
    for models, sizes in ((["M1", "M2"], [500]), (["M3", "M4"], range(100, 501, 100))):
        for n in sizes:
            blocks = [(model, tables[model, n]) for model in models if (model, n) in tables]
            if not blocks:
                continue
            name = "_".join(model for model, _ in blocks) + (f"_n{n}" if models[0] == "M3" else "")
            tex = latex_table(blocks, f"RE by dimension $d$, $n={n}$ and $N=5000$.")
            (output / "latex" / f"{name}.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", action="store_true", help="use results/reference/csv")
    source.add_argument("--runs", nargs="+", type=Path, help="output folders of finished runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tables = read_reference() if args.reference else read_runs(args.runs)
    default = "reference_tables" if args.reference else "recomputed_tables"
    output = args.output or ROOT / "outputs" / default
    export(tables, output)
    print(f"{len(tables)} tables written to {output.resolve()}")


if __name__ == "__main__":
    main()
