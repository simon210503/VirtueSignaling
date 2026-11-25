from simulation import run_simulation
from plotting import (
    plot_strategy_dynamics,
    plot_reputation_dynamics
)


def main():

    solution, params = run_simulation()

    plot_strategy_dynamics(solution, params)
    plot_reputation_dynamics(solution, params)


if __name__ == "__main__":
    main()
