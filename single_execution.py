
from model_generations import run_simulation, save_results

from plotting import (
    plot_strategy_dynamics,
    plot_reputation_dynamics,
    plot_round_reputations
)


def main():

    # Simulation ausführen
    result, params = run_simulation()

    # Ergebnisse speichern
    save_results(
        result,
        params,
        result["runtime_seconds"]
    )

    # Plots erstellen
    plot_strategy_dynamics(result, params)
    plot_reputation_dynamics(result, params)
    plot_round_reputations(result, params)


if __name__ == "__main__":
    main()
