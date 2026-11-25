
import json
import copy
import itertools
import csv
from pathlib import Path
from datetime import datetime

from model_generations import run_simulation


# ==================================================
# GRUNDEINSTELLUNGEN
# ==================================================

CONFIG_FILE = Path("config.json")
RESULTS_DIR = Path("sweep_results")


# ==================================================
# PARAMETERVARIATIONEN
# ==================================================

def define_parameter_variations():

    variations = {

        # Modellparameter
        ("parameters", "b"): [2.0, 3.0, 4.0],
        ("parameters", "c"): [1.0],
        ("parameters", "s"): [0.25, 0.5, 0.75],
        ("parameters", "q"): [0.1, 0.5, 0.7, 0.9, 1.0],

        # Simulationsparameter
        ("simulation", "selection_strength"): [
            1e-4, 1e-3, 1e-2, 1e-1
        ],

        # Anfangsbedingungen: Shares
        ("strategies", "ALLD", "share"): [
            1,2,3,10,100
        ],

        ("strategies", "DISC", "share"): [
            1,2,3,10,100
        ],

        ("strategies", "ALLD_VS", "share"): [
            1,2,3,10,100
        ],

        # Anfangsbedingungen: Reputationen
        ("strategies", "ALLD", "reputation"): [
            0.0, 0.5, 1.0
        ],

        ("strategies", "DISC", "reputation"): [
            0.0, 0.5, 1.0
        ],

        ("strategies", "ALLD_VS", "reputation"): [
            0.0, 0.5, 1.0
        ]
    }

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


# ==================================================
# PARAMETER-SWEEP
# ==================================================

def run_parameter_sweep():

    base_config = load_config()
    variations = define_parameter_variations()

    parameter_names = list(variations.keys())
    parameter_values = list(variations.values())

    combinations = itertools.product(*parameter_values)

    n_combinations = 1

    for values in parameter_values:
        n_combinations *= len(values)

    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    print(f"Number of simulations: {n_combinations}")

    # Ergebnisordner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporäre Konfigurationsdatei
    temp_config = output_dir / "temp_config.json"

    # CSV-Datei für Zusammenfassung
    csv_file = output_dir / "summary.csv"

    fieldnames = [
        "run"
    ] + [
        make_parameter_label(name)
        for name in parameter_names
    ] + [
        "runtime"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Simulationen ausführen
        for run_number, combination in enumerate(combinations, start=1):

            config = copy.deepcopy(base_config)

            # Parameterkombination einsetzen
            current_values = dict(zip(parameter_names, combination))

            for name, value in current_values.items():
                set_nested_value(config, name, value)

            # Konfiguration speichern
            with open(temp_config, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=4)

            print(
                f"\nSimulation {run_number}/{n_combinations}"
            )

            # Simulation starten
            try:
                result, params = run_simulation(
                    config_path=str(temp_config)
                )

                # Laufzeit aus der Simulation übernehmen,
                # sofern sie in params gespeichert ist.
                row = {"run": run_number}

                for name, value in current_values.items():
                    row[make_parameter_label(name)] = value

                writer.writerow(row)
                file.flush()

            except Exception as error:

                print(f"Error in simulation {run_number}: {error}")

                row = {"run": run_number}

                for name, value in current_values.items():
                    row[make_parameter_label(name)] = value

                row["runtime"] = "ERROR"

                writer.writerow(row)
                file.flush()

    # Temporäre Datei entfernen
    if temp_config.exists():
        temp_config.unlink()

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP COMPLETED")
    print("=" * 60)
    print(f"Results saved in: {output_dir}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    run_parameter_sweep()
