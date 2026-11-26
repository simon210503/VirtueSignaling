"""
Auswertung eines Parameter-Sweeps (summary.csv).

Aufruf:
    python analyze_sweep.py                          # neuester Sweep in sweep_results/
    python analyze_sweep.py sweep_results/<ordner>/summary.csv
    python analyze_sweep.py --filter q=0.5,0.9 b=2   # nur Teilmenge auswerten
    python analyze_sweep.py --heatmap q s            # Karte über zwei Parameter
    python analyze_sweep.py --heatmap q s --heatmap b q
    python analyze_sweep.py --no-plots               # nur Tabellen

Ergebnisse: Ausgabe in der Konsole, dazu Tabellen (CSV), Bericht (report.txt)
und Diagramme (PNG) im Ordner  <ordner der summary.csv>/analysis/ .

Begriffe:
    result = Gewinner, falls outcome == dominance (z. B. "DISC"),
             sonst der Ausgang selbst ("coexistence", "not_converged", "ERROR").
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_COLUMNS = {
    "run", "outcome", "winner", "coexisting_strategies",
    "convergence_generation", "runtime_seconds",
}

# Reihenfolge der Nicht-Gewinner-Kategorien in Tabellen
SPECIAL_RESULTS = ["coexistence", "not_converged", "ERROR"]

# Farben (Okabe-Ito, farbenblindfreundlich) für Strategien; feste Farben für den Rest
STRATEGY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
SPECIAL_COLORS = {
    "coexistence": "#F0E442",
    "not_converged": "#999999",
    "ERROR": "#000000",
}

NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


# ==================================================
# EINLESEN
# ==================================================

def find_latest_summary(root=Path("sweep_results")):

    files = sorted(root.glob("*/summary.csv"))   # Ordnernamen sind Zeitstempel

    if not files:
        sys.exit(f"Keine summary.csv unter '{root}/' gefunden. Pfad als Argument angeben.")

    return files[-1]


def auto_numeric(series):
    """Spalte in Zahlen umwandeln, wenn ALLE Werte Zahlen sind (sonst Text lassen)."""

    values = series.astype(str)

    if values.map(lambda v: bool(NUMBER_PATTERN.match(v))).all():
        return pd.to_numeric(values)

    return series


def coexisting_label(text):

    try:
        strategies = json.loads(text)
    except (TypeError, ValueError):
        return ""

    return "+".join(strategies) if strategies else ""


def load_summary(path):

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    parameters = [c for c in df.columns if c not in RESULT_COLUMNS]

    for column in parameters:
        df[column] = auto_numeric(df[column])

    for column in ["run", "convergence_generation", "runtime_seconds"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["coexisting"] = df["coexisting_strategies"].map(coexisting_label)

    df["result"] = np.where(
        (df["outcome"] == "dominance") & (df["winner"] != ""),
        df["winner"],
        df["outcome"],
    )

    return df, parameters


# ==================================================
# SPALTEN / FILTER
# ==================================================

def resolve_column(name, columns):
    """'q' findet 'parameters.q'; ein vollständiger Name geht immer."""

    if name in columns:
        return name

    matches = [c for c in columns if c.endswith("." + name)]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        sys.exit(f"Spalte '{name}' nicht gefunden. Verfügbar: {', '.join(columns)}")

    sys.exit(f"Spalte '{name}' ist mehrdeutig: {', '.join(matches)}")


def apply_filters(df, parameters, filters):

    for expression in filters or []:

        if "=" not in expression:
            sys.exit(f"Filter '{expression}' ungültig. Format: name=wert1,wert2")

        name, raw_values = expression.split("=", 1)
        column = resolve_column(name, parameters)
        wanted = [v.strip() for v in raw_values.split(",")]

        if pd.api.types.is_numeric_dtype(df[column]):
            mask = np.zeros(len(df), dtype=bool)
            for value in wanted:
                mask |= np.isclose(df[column], float(value))
        else:
            mask = df[column].astype(str).isin(wanted)

        df = df[mask]

    return df


def short(name):
    """Anzeigename ohne 'parameters.'-Präfix."""

    return name.replace("parameters.", "")


# ==================================================
# TABELLEN
# ==================================================

def result_order(df):

    strategies = sorted(
        r for r in df["result"].unique() if r not in SPECIAL_RESULTS
    )
    special = [r for r in SPECIAL_RESULTS if r in set(df["result"])]

    return strategies + special


def count_table(df, column):
    """Zeilen = Parameterwerte, Spalten = Ergebnisse, plus Gesamtzahl n."""

    counts = pd.crosstab(df[column], df["result"])
    counts = counts.reindex(columns=result_order(df), fill_value=0)
    counts["n"] = counts.sum(axis=1)
    counts.index.name = short(column)
    counts.columns.name = None

    return counts


def percent_table(counts):

    percent = counts.drop(columns="n").div(counts["n"], axis=0) * 100

    return percent


def combined_table(counts):
    """'Anzahl (Prozent)' als Text, für die Konsolenausgabe."""

    percent = percent_table(counts)
    shown = pd.DataFrame(index=counts.index)

    for column in percent.columns:
        shown[column] = (
            counts[column].astype(str)
            + " ("
            + percent[column].round().astype(int).astype(str)
            + "%)"
        )

    shown["n"] = counts["n"]

    return shown


# ==================================================
# AUSGABE-HILFEN
# ==================================================

class Report:
    """Schreibt in die Konsole und sammelt denselben Text für report.txt."""

    def __init__(self):
        self.lines = []

    def text(self, content=""):
        print(content)
        self.lines.append(str(content))

    def section(self, title):
        self.text("\n" + "=" * 70)
        self.text(title)
        self.text("=" * 70)

    def table(self, frame):
        self.text(frame.to_string())

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


# ==================================================
# AUSWERTUNGEN
# ==================================================

def overview(df, report, out_dir):

    report.section("OVERVIEW")

    n = len(df)
    n_error = int((df["outcome"] == "ERROR").sum())

    report.text(f"Simulations:           {n}")
    report.text(f"Errors:                {n_error}")

    total_runtime = df["runtime_seconds"].sum()
    report.text(
        f"Runtime (sum over runs): {total_runtime:,.1f} s "
        f"({total_runtime / 3600:.2f} h CPU time)"
    )
    report.text(f"Mean runtime per run:  {df['runtime_seconds'].mean():.2f} s")

    # --- Ausgang ---
    report.text("\nOutcome:")
    outcome = df["outcome"].value_counts()
    outcome = pd.DataFrame({
        "count": outcome,
        "percent": (outcome / n * 100).round(1),
    })
    report.table(outcome)
    outcome.to_csv(out_dir / "overview_outcome.csv")

    # --- Gewinner / Ergebnis ---
    report.text("\nResult (winner of a clear victory, otherwise the outcome):")
    result = df["result"].value_counts().reindex(result_order(df))
    result = pd.DataFrame({
        "count": result,
        "percent": (result / n * 100).round(1),
    })
    report.table(result)
    result.to_csv(out_dir / "overview_result.csv")

    # --- Koexistenz ---
    coexist = df[df["outcome"] == "coexistence"]

    if len(coexist):
        report.text("\nCoexisting strategy sets (outcome = coexistence):")
        sets = coexist["coexisting"].value_counts().rename("count")
        report.table(sets.to_frame())
        sets.to_csv(out_dir / "overview_coexistence_sets.csv")

        strategies = sorted(set("+".join(coexist["coexisting"]).split("+")) - {""})
        involvement = pd.Series({
            s: int(coexist["coexisting"].map(lambda v: s in v.split("+")).sum())
            for s in strategies
        }, name="runs_in_coexistence")
        report.text("\nRuns in which a strategy is part of the coexistence:")
        report.table(involvement.to_frame())

    # --- Konvergenz ---
    converged = df[df["outcome"].isin(["dominance", "coexistence"])]

    if len(converged):
        report.text("\nConvergence generation (only converged runs), by result:")
        stats = converged.groupby("result")["convergence_generation"].agg(
            ["count", "median", "mean", "max"]
        ).round(0)
        report.table(stats)
        stats.to_csv(out_dir / "overview_convergence_by_result.csv")

    # --- Laufzeit ---
    report.text("\nRuntime by outcome (where does the compute time go?):")
    runtime = df.groupby("outcome")["runtime_seconds"].agg(["count", "sum", "mean", "max"])
    runtime["share_of_total_%"] = (runtime["sum"] / total_runtime * 100).round(1)
    report.table(runtime.round(2))
    runtime.to_csv(out_dir / "overview_runtime_by_outcome.csv")


def slowest_runs(df, parameters, report, out_dir, top):

    report.section(f"{top} SLOWEST RUNS")

    columns = ["run"] + parameters + ["result", "convergence_generation", "runtime_seconds"]
    slow = df.sort_values("runtime_seconds", ascending=False).head(top)[columns]

    report.table(slow.rename(columns=short).round(2).set_index("run"))
    slow.to_csv(out_dir / "slowest_runs.csv", index=False)


def by_parameter(df, parameters, report, out_dir):

    varied = [p for p in parameters if df[p].nunique() > 1]
    fixed = [p for p in parameters if df[p].nunique() <= 1]

    if fixed:
        report.text(
            "\nConstant columns (not analysed): "
            + ", ".join(f"{short(p)}={df[p].iloc[0]}" for p in fixed)
        )

    for column in varied:

        report.section(f"BY PARAMETER: {short(column)}   (count and percent of runs)")

        counts = count_table(df, column)
        report.table(combined_table(counts))

        stem = slug(short(column))
        counts.to_csv(out_dir / f"by_{stem}_counts.csv")
        percent_table(counts).round(2).to_csv(out_dir / f"by_{stem}_percent.csv")

        converged = df[df["outcome"].isin(["dominance", "coexistence"])]

        if len(converged):
            median = converged.pivot_table(
                index=column, columns="result",
                values="convergence_generation", aggfunc="median",
            )
            report.text(f"\nMedian convergence generation by {short(column)}:")
            report.table(median.round(0))
            median.to_csv(out_dir / f"by_{stem}_median_convergence.csv")

    return varied


# ==================================================
# DIAGRAMME
# ==================================================

def color_map(df):

    colors = dict(SPECIAL_COLORS)
    strategies = sorted(r for r in df["result"].unique() if r not in SPECIAL_RESULTS)

    for i, strategy in enumerate(strategies):
        colors[strategy] = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]

    return colors


def plot_stacked_bars(df, varied, out_dir):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = color_map(df)

    for column in varied:

        counts = count_table(df, column)
        percent = percent_table(counts)

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(percent) + 2), 5))
        bottom = np.zeros(len(percent))
        labels = [f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)
                  for v in percent.index]

        for result in percent.columns:
            ax.bar(labels, percent[result], bottom=bottom,
                   label=result, color=colors[result], edgecolor="white", linewidth=0.5)
            bottom += percent[result].to_numpy()

        for i, total in enumerate(counts["n"]):
            ax.text(i, 101, f"n={int(total)}", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0, 108)
        ax.set_ylabel("Share of runs [%]")
        ax.set_xlabel(short(column))
        ax.set_title(f"Result by {short(column)}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

        if max(len(l) for l in labels) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig.tight_layout()
        fig.savefig(out_dir / f"plot_{slug(short(column))}.png", dpi=150)
        plt.close(fig)


def plot_heatmaps(df, column_x, column_y, out_dir, report):
    """Anteil jedes Ergebnisses je Zelle (gemittelt über alle anderen Parameter)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = result_order(df)
    n_cells = df.groupby([column_y, column_x]).size().unstack(column_x)

    report.section(
        f"HEATMAP: {short(column_y)} (rows) x {short(column_x)} (columns)"
        f"   most frequent result per cell"
    )

    # Textuelle Karte: häufigstes Ergebnis und sein Anteil
    mode = df.groupby([column_y, column_x])["result"].agg(
        lambda s: f"{s.value_counts().index[0]} ({s.value_counts().iloc[0] / len(s):.0%})"
    ).unstack(column_x)
    report.table(mode)
    mode.to_csv(out_dir / f"heatmap_{slug(short(column_x))}_{slug(short(column_y))}_mode.csv")

    fig, axes = plt.subplots(
        1, len(results), figsize=(3.6 * len(results) + 1, 3.8), squeeze=False
    )

    for ax, result in zip(axes[0], results):

        share = (
            df.assign(hit=(df["result"] == result).astype(float))
            .groupby([column_y, column_x])["hit"].mean()
            .unstack(column_x)
        )
        share = share.reindex(index=n_cells.index, columns=n_cells.columns)

        image = ax.imshow(share.to_numpy(), vmin=0, vmax=1, cmap="viridis",
                          origin="lower", aspect="auto")

        ax.set_xticks(range(len(share.columns)))
        ax.set_xticklabels([f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)
                            for v in share.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(share.index)))
        ax.set_yticklabels([f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)
                            for v in share.index])
        ax.set_xlabel(short(column_x))
        ax.set_title(result)

        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                value = share.iat[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if value < 0.6 else "black")

    axes[0][0].set_ylabel(short(column_y))
    fig.colorbar(image, ax=axes[0].tolist(), label="Share of runs", shrink=0.9)

    filename = f"heatmap_{slug(short(column_x))}_{slug(short(column_y))}.png"
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser(description="Auswertung eines Parameter-Sweeps")
    parser.add_argument("summary", nargs="?", help="Pfad zur summary.csv (Standard: neuester Sweep)")
    parser.add_argument("--filter", nargs="+", metavar="NAME=W1,W2",
                        help="Nur Läufe mit diesen Parameterwerten auswerten")
    parser.add_argument("--heatmap", nargs=2, action="append", metavar=("X", "Y"),
                        help="Karte über zwei Parameter (mehrfach angebbar)")
    parser.add_argument("--no-plots", action="store_true", help="Keine Diagramme erzeugen")
    parser.add_argument("--top", type=int, default=10, help="Anzahl der langsamsten Läufe (Standard 10)")
    parser.add_argument("--out", help="Ausgabeordner (Standard: <summary-ordner>/analysis)")
    args = parser.parse_args()

    path = Path(args.summary) if args.summary else find_latest_summary()

    if not path.exists():
        sys.exit(f"Datei nicht gefunden: {path}")

    df, parameters = load_summary(path)

    if args.filter:
        total = len(df)
        df = apply_filters(df, parameters, args.filter)
        print(f"Filter {args.filter}: {len(df)} of {total} runs remain.")

    if df.empty:
        sys.exit("Keine Läufe nach Anwendung der Filter übrig.")

    out_dir = Path(args.out) if args.out else path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    report = Report()
    report.text(f"File: {path}")
    if args.filter:
        report.text(f"Filter: {' '.join(args.filter)}")

    overview(df, report, out_dir)
    slowest_runs(df, parameters, report, out_dir, args.top)
    varied = by_parameter(df, parameters, report, out_dir)

    if not args.no_plots:
        plot_stacked_bars(df, varied, out_dir)

    for name_x, name_y in args.heatmap or []:
        column_x = resolve_column(name_x, parameters)
        column_y = resolve_column(name_y, parameters)
        if not args.no_plots:
            plot_heatmaps(df, column_x, column_y, out_dir, report)

    report.save(out_dir / "report.txt")

    print("\n" + "=" * 70)
    print(f"Tables, report and plots saved in: {out_dir}")


if __name__ == "__main__":
    main()