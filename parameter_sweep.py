import os

# Jeder Worker soll genau einen Kern nutzen (sonst starten NumPy/BLAS
# zusätzliche Threads und die Prozesse bremsen sich gegenseitig aus).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import copy
import itertools
import csv
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime

from model_generations import run_simulation, warm_up


# ==================================================
# GRUNDEINSTELLUNGEN
# ==================================================

CONFIG_FILE = Path("config.json")
RESULTS_DIR = Path("sweep_results")

# Anzahl paralleler Prozesse. None = alle Kerne außer einem (der Laptop
# bleibt bedienbar). Auf dem Server z. B. 32 eintragen.
N_WORKERS = None


# ==================================================
# PARAMETERVARIATIONEN
# ==================================================
#
# So gehst du vor:
#   1. In CANDIDATES die Werte eintragen, die getestet werden können.
#   2. In VARY die Parameter ein- bzw. auskommentieren, die variiert werden
#      sollen. Alles, was nicht in VARY steht, bleibt auf dem Wert aus
#      config.json.
#   3. Vor dem Start wird die Zahl der Simulationen ausgegeben. Übersteigt sie
#      MAX_COMBINATIONS, bricht das Skript ab (Schutz vor Versehen: das
#      vollständige Gitter wäre mehrere zehn Millionen Läufe groß).

MAX_COMBINATIONS = 500_000


# Reputationsmechanismus: None steht für "*" (0 oder 1). Es werden alle
# Kombinationen erzeugt, hier also 2 * 2 * 2 = 8 Mechanismen.
MECHANISM_TEMPLATE = [
    [1, None, 1, None],
    [0, 1,    0, None],
]


def expand_mechanism_template(template):

    stars = [
        (i, j)
        for i, row in enumerate(template)
        for j, value in enumerate(row)
        if value is None
    ]

    mechanisms = []

    for bits in itertools.product([0, 1], repeat=len(stars)):
        mechanism = copy.deepcopy(template)
        for (i, j), bit in zip(stars, bits):
            mechanism[i][j] = bit
        mechanisms.append(mechanism)

    return mechanisms


# Alle Kandidatenwerte (hier ändern/erweitern)
CANDIDATES = {

    # --- Modellparameter ---
    "b": [2, 3, 4, 5, 10],
    "c": [0.5, 1.0, 2.0],
    "s": [0.01, 0.1, 0.25, 0.5, 0.7, 0.8, 0.9],
    "q": [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0],

    # --- Reputationsmechanismus ---
    "reputation_mechanism": expand_mechanism_template(MECHANISM_TEMPLATE),

    # --- Startbedingungen (gelten je für JEDE aktive Strategie, unabhängig) ---
    "strategy_share": [1, 2, 3, 5, 10, 100],
    "strategy_reputation": [0, 0.5, 1],

    # --- Simulationsparameter ---
    "n_rounds": [1, 5, 10, 20, 50, 100],
    "selection_strength": [1e-2, 1e-1, 1.0],      # Schrittweite pro Generation (mittlere Auszahlung pro Runde)
    "reputation_step": [0.25, 0.5, 1.0],           # 1.0 = volle Ersetzung, 0.5 = wie im Paper
    "reset_reputations": [True, False],            # Reputationen je Generation zurücksetzen?
    "n_generations": [10_000, 100_000],            # Obergrenze (Einfluss auf "not_converged")
}


# Welche Parameter variiert werden. Zeile ein-/auskommentieren genügt.
VARY = [
    # "b",
    # "s",
     "q",
    # "c",
     "reputation_mechanism",
     "strategy_share",
     "strategy_reputation",
    # "n_rounds",
    # "selection_strength",
    # "reputation_step",
    # "reset_reputations",
    # "n_generations",
]


def _targets(name, active_strategies):
    """Pfade in der Config, die ein Parameter setzt."""

    simple = {
        "b": [("parameters", "b")],
        "c": [("parameters", "c")],
        "s": [("parameters", "s")],
        "q": [("parameters", "q")],
        "reputation_mechanism": [("reputation_mechanism",)],
        "n_rounds": [("simulation", "n_rounds")],
        "selection_strength": [("simulation", "selection_strength")],
        "reputation_step": [("simulation", "reputation_step")],
        "reset_reputations": [("simulation", "reset_reputations")],
        "n_generations": [("simulation", "n_generations")],
    }

    if name in simple:
        return simple[name]

    if name == "strategy_share":
        return [("strategies", s, "share") for s in active_strategies]

    if name == "strategy_reputation":
        return [("strategies", s, "reputation") for s in active_strategies]

    raise ValueError(f"Unbekannter Parameter in VARY: '{name}'")


def define_parameter_variations():

    unknown = [name for name in VARY if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"In VARY ohne Eintrag in CANDIDATES: {unknown}")

    # Aktive Strategien direkt aus der config.json
    config = load_config()
    active_strategies = [
        name for name, settings in config["strategies"].items()
        if settings["enabled"]
    ]

    variations = {}

    for name in VARY:
        for keys in _targets(name, active_strategies):
            variations[keys] = CANDIDATES[name]

    # Übersicht und Schutz vor versehentlich riesigen Sweeps
    total = 1
    print("=" * 60)
    print("PARAMETER GRID")
    print("=" * 60)

    for keys, values in variations.items():
        total *= len(values)
        print(f"  {make_parameter_label(keys):<40} {len(values):>3} values")

    if not variations:
        print("  (nothing varied - a single simulation with config.json)")

    print(f"  {'Total':<40} {total} simulations")

    if total > MAX_COMBINATIONS:
        raise ValueError(
            f"{total:,} Simulationen überschreiten MAX_COMBINATIONS = "
            f"{MAX_COMBINATIONS:,}. Weniger Parameter in VARY aktivieren, "
            f"Kandidatenlisten kürzen oder MAX_COMBINATIONS erhöhen."
        )

    return variations


# ==================================================
# HILFSFUNKTIONEN
# ==================================================

def load_config():

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def set_nested_value(dictionary, keys, value):

    current = dictionary

    for key in keys[:-1]:
        current = current[key]

    current[keys[-1]] = value


def get_nested_value(dictionary, keys):

    current = dictionary

    for key in keys:
        current = current[key]

    return current


def make_parameter_label(keys):

    return ".".join(keys)


def format_parameter_value(value):
    """
    Wert für die CSV. Eine Matrix (z. B. der Reputationsmechanismus) wird
    kompakt als Zeilen von Ziffern geschrieben:
        [[1, 0, 1, 0], [0, 1, 0, 0]]  ->  "1010_0100"
    """
    if isinstance(value, list) and value and all(isinstance(r, list) for r in value):
        return "_".join("".join(str(v) for v in row) for row in value)

    return value


# ==================================================
# EINZELNER LAUF (läuft in einem Worker-Prozess)
# ==================================================

def run_single(run_number, config):
    """Gibt (run_number, ergebnis_dict, fehlertext) zurück."""

    try:
        # Config direkt als Dictionary: keine gemeinsame temp-Datei nötig,
        # und nur der Endzustand wird gespeichert (schnell, wenig RAM).
        result, _ = run_simulation(
            config=config,
            record_history=False,
            verbose=False
        )

        row = {
            "outcome": result.get("outcome"),
            "winner": result.get("winner"),
            "coexisting_strategies": json.dumps(
                result.get("coexisting_strategies"),
                ensure_ascii=False
            ),
            "convergence_generation": result.get("convergence_generation"),
            "runtime_seconds": result.get("runtime_seconds"),
        }

        return run_number, row, None

    except Exception:

        return run_number, None, traceback.format_exc()


# ==================================================
# PARAMETER-SWEEP
# ==================================================

def run_parameter_sweep():

    base_config = load_config()
    variations = define_parameter_variations()

    parameter_names = list(variations.keys())
    parameter_values = list(variations.values())

    n_combinations = 1
    for values in parameter_values:
        n_combinations *= len(values)

    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, n_combinations)

    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    print(f"Number of simulations: {n_combinations}")
    print(f"Parallel workers:      {n_workers}")

    # Ergebnisordner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "summary.csv"

    fieldnames = (
        ["run"]
        + [make_parameter_label(name) for name in parameter_names]
        + [
            "outcome",
            "winner",
            "coexisting_strategies",
            "convergence_generation",
            "runtime_seconds"
        ]
    )

    # Numba einmal im Hauptprozess kompilieren (bzw. aus dem Cache laden),
    # damit nicht jeder Worker für sich kompiliert.
    print("\nPreparing simulation core (first start may take a few seconds)...")
    warm_up()

    combinations = enumerate(itertools.product(*parameter_values), start=1)

    all_rows = []
    param_by_run = {}
    start_time = time.perf_counter()
    n_done = 0

    def make_row(run_number, result_row, error):
        row = {"run": run_number}
        row.update(param_by_run.pop(run_number))

        if error is None:
            row.update(result_row)
        else:
            print(f"\nError in simulation {run_number}:\n{error}")
            row.update({
                "outcome": "ERROR",
                "winner": None,
                "coexisting_strategies": None,
                "convergence_generation": None,
                "runtime_seconds": None,
            })
        return row

    with open(csv_file, "w", newline="", encoding="utf-8") as file, \
            ProcessPoolExecutor(max_workers=n_workers) as executor:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        pending = set()

        def submit_next():
            # Nur eine begrenzte Zahl Aufgaben gleichzeitig vorhalten, damit auch
            # sehr große Sweeps nicht alle Configs auf einmal im Speicher halten.
            item = next(combinations, None)
            if item is None:
                return False

            run_number, combination = item
            config = copy.deepcopy(base_config)

            values = dict(zip(parameter_names, combination))
            for name, value in values.items():
                set_nested_value(config, name, value)

            param_by_run[run_number] = {
                make_parameter_label(name): format_parameter_value(value)
                for name, value in values.items()
            }

            pending.add(executor.submit(run_single, run_number, config))
            return True

        try:
            for _ in range(2 * n_workers):
                if not submit_next():
                    break

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    pending.discard(future)
                    run_number, result_row, error = future.result()

                    row = make_row(run_number, result_row, error)
                    writer.writerow(row)
                    file.flush()
                    all_rows.append(row)

                    n_done += 1
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[{n_done}/{n_combinations}] run {run_number}: "
                        f"{row['outcome']}  ({elapsed:.0f}s elapsed)"
                    )

                    submit_next()

        except KeyboardInterrupt:
            print("\nInterrupted - cancelling remaining simulations...")
            for future in pending:
                future.cancel()
            raise

    # Datei am Ende nach Laufnummer sortieren (Ergebnisse treffen unsortiert ein)
    all_rows.sort(key=lambda r: r["run"])
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP COMPLETED")
    print("=" * 60)
    print(f"Total time: {total:.1f}s")
    print(f"Results saved in: {output_dir}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    run_parameter_sweep()