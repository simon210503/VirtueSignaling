import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path


from itertools import product

# ============================================================
# Simulation settings
# ============================================================

RUN_MODE = "sweep"  # Options: "test" or "sweep"

BASE_PARAMS = {
    "b": 2.0,
    "c": 1.0,
    "s": 0.6,
    "g_x": 1,
    "g_y": 0,
    "g_z": 1,
    "g_v": 1
}

# Initial conditions for testing
TEST_INITIAL = (0.25, 0.25, 0.25)


# ============================================================
# Parameter sweep settings
# ============================================================

PARAMETER_GRID = {
    "b": [1.5, 2.0, 3.0, 5.0, 10.0, 100.0],
    "c": [1.0],
    "s": [0.1, 0.25, 0.5, 0.75, 0.9]
}

# Different initial conditions
INITIAL_CONDITIONS = [
    (0.25, 0.25, 0.25),
    (0.7, 0.1, 0.1),
    (0.1, 0.7, 0.1),
    (0.1, 0.1, 0.7),
    (0.1, 0.1, 0.1)
]


def payoff(x, y, z, v, params):

    b = params["b"]
    c = params["c"]
    s = params["s"]

    g_x = params["g_x"]
    g_y = params["g_y"]
    g_z = params["g_z"]
    g_v = params["g_v"]

    g = x * g_x + y * g_y + z * g_z + v * g_v

    P_x = b * (x + z * g_x) - c
    P_y = b * (x + z * g_y)
    P_z = b * (x + z * g_z) - c * g
    P_v = b * (x + z * g_v) - s

    P_bar = x * P_x + y * P_y + z * P_z + v * P_v

    return g, P_x, P_y, P_z, P_v, P_bar


def replicator_eq(t, current_state, params):

    x, y, z = current_state
    v = 1.0 - x - y - z

    g, P_x, P_y, P_z, P_v, P_bar = payoff(
        x, y, z, v, params
    )

    dx = x * (P_x - P_bar)
    dy = y * (P_y - P_bar)
    dz = z * (P_z - P_bar)

    return [dx, dy, dz]


def convergence_event(t, current_state, params):

    derivatives = replicator_eq(t, current_state, params)

    return np.max(np.abs(derivatives)) - 1e-8


convergence_event.terminal = True
convergence_event.direction = -1

def run_simulation(params, initial_conditions):

    t_start = 0.0
    t_end = 1000.0
    t_points = 150000

    y0_vector = np.array(initial_conditions, dtype=float)

    if np.any(y0_vector < 0) or np.any(y0_vector > 1):
        raise ValueError("Initial fractions must be between 0 and 1.")

    if np.sum(y0_vector) > 1:
        raise ValueError("Initial fractions must sum to 1.")

    t_eval = np.linspace(t_start, t_end, t_points)

    solution = solve_ivp(
        fun=replicator_eq,
        t_span=(t_start, t_end),
        y0=y0_vector,
        t_eval=t_eval,
        args=(params,),
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        events=[convergence_event]
    )

    # Extract results
    t = solution.t
    x, y, z = solution.y
    v = 1.0 - x - y - z
    
    # Print final strategy shares
    print("\nFinal strategy shares:")
    print(f"x(T) = {x[-1]:.6f}")
    print(f"y(T) = {y[-1]:.6f}")
    print(f"z(T) = {z[-1]:.6f}")
    print(f"v(T) = {v[-1]:.6f}")

    # Check whether strategy shares remain valid
    total = np.sum(solution.y, axis=0)

    if (
        np.any(np.array(solution.y) < -1e-6)
        or np.any(np.array(solution.y) > 1 + 1e-6)
        or np.any(v < -1e-6)
        or np.any(v > 1 + 1e-6)
    ):
        raise RuntimeError("Strategy shares outside valid range!")

    if not solution.success:
        raise RuntimeError(solution.message)

    # Check why the simulation stopped
    if len(solution.t_events[0]) > 0:
        print("Convergence reached.")

    else:
        print("Maximum simulation time reached.")

    return solution

def plot_solution(solution, params, initial_conditions):

    t = solution.t
    x, y, z = solution.y
    v = 1 - x - y - z

    # Create parameter-specific directory
    parameter_folder = f"b={params['b']:g}_s={params['s']:g}"

    save_dir = Path("results") / parameter_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create filename based on initial conditions
    x0, y0, z0 = initial_conditions
    v0 = 1 - x0 - y0 - z0

    filename = (
        f"IC_x={x0:g}_y={y0:g}_z={z0:g}_v={v0:g}.png"
    )

    filepath = save_dir / filename

    # Create plot
    plt.figure(figsize=(10, 6))

    plt.plot(t, x, label="ALLC", linewidth=2)
    plt.plot(t, y, label="ALLD", linewidth=2)
    plt.plot(t, z, label="Discriminator", linewidth=2)
    plt.plot(t, v, label="Pure Virtue Signaling", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Strategy share")
    plt.title("Replicator Dynamics")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {filepath}")

    #plt.show()
    plt.close()


def run_single_test():

    print("Running single test...")

    solution = run_simulation(
        BASE_PARAMS,
        TEST_INITIAL
    )

    plot_solution(solution, BASE_PARAMS, TEST_INITIAL)



def run_parameter_sweep():

    print("Running parameter sweep...")

    parameter_names = list(PARAMETER_GRID.keys())
    parameter_values = list(PARAMETER_GRID.values())

    results = []
    failed_simulations = []

    for values in product(*parameter_values):

        params = BASE_PARAMS.copy()

        for name, value in zip(parameter_names, values):
            params[name] = value

        for initial in INITIAL_CONDITIONS:

            try:
                solution = run_simulation(params, initial)

                plot_solution(solution, params, initial)

                x, y, z = solution.y
                v = 1 - x - y - z

                final_shares = np.array([x[-1], y[-1], z[-1], v[-1]])

                strategy_names = [
                    "ALLC",
                    "ALLD",
                    "Discriminator",
                    "Pure Virtue Signaling"
                ]

                winner = strategy_names[np.argmax(final_shares)]

                results.append({
                    "params": params.copy(),
                    "initial": initial,
                    "final_x": x[-1],
                    "final_y": y[-1],
                    "final_z": z[-1],
                    "final_v": v[-1],
                    "winner": winner
                })

                print(
                    f"b={params['b']}, "
                    f"c={params['c']}, "
                    f"s={params['s']} | "
                    f"Initial={initial} | "
                    f"Final={[round(x[-1], 3), round(y[-1], 3), round(z[-1], 3), round(v[-1], 3)]}"
                )

            except (RuntimeError, ValueError) as error:

                print("\nSimulation failed. Skipping...")
                print(
                    f"b={params['b']}, "
                    f"c={params['c']}, "
                    f"s={params['s']} | "
                    f"Initial={initial}"
                )
                print(f"Error: {error}")

                failed_simulations.append({
                    "params": params.copy(),
                    "initial": initial,
                    "error": str(error)
                })

                continue

    print("\nParameter sweep completed.")
    print(f"Successful simulations: {len(results)}")
    print(f"Failed simulations: {len(failed_simulations)}")

    return results, failed_simulations


def plot_winner_map(results):

    strategy_names = [
        "ALLC",
        "ALLD",
        "Discriminator",
        "Pure Virtue Signaling"
    ]

    # Special categories
    categories = strategy_names + ["Mixed", "No data"]

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:purple",
        "gold",
        "lightgray"
    ]

    # Get sorted parameter values
    b_values = sorted(PARAMETER_GRID["b"])
    s_values = sorted(PARAMETER_GRID["s"])

    # Initialise grid
    winner_grid = np.full(
        (len(s_values), len(b_values)),
        len(categories) - 1
    )

    # Determine the winner for each parameter combination
    for i, s in enumerate(s_values):
        for j, b in enumerate(b_values):

            # Select all successful simulations for this b and s
            matching_results = [
                r for r in results
                if r["params"]["b"] == b
                and r["params"]["s"] == s
            ]

            if not matching_results:
                continue

            # Count how often each strategy won
            winner_counts = {
                strategy: sum(
                    r["winner"] == strategy
                    for r in matching_results
                )
                for strategy in strategy_names
            }

            max_wins = max(winner_counts.values())

            # Check for ties
            winners = [
                strategy
                for strategy, count in winner_counts.items()
                if count == max_wins
            ]

            if len(winners) == 1:
                winner_grid[i, j] = strategy_names.index(winners[0])
            else:
                winner_grid[i, j] = len(strategy_names)

    # Create plot
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(categories) + 0.5, 1),
        cmap.N
    )

    # Calculate cell boundaries so unequal parameter intervals
    # are represented proportionally.
    def get_edges(values):
        values = np.asarray(values, dtype=float)

        if len(values) == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])

        midpoints = (values[:-1] + values[1:]) / 2

        first = values[0] - (midpoints[0] - values[0])
        last = values[-1] + (values[-1] - midpoints[-1])

        return np.concatenate(([first], midpoints, [last]))

    b_edges = get_edges(b_values)
    s_edges = get_edges(s_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    mesh = ax.imshow(
        winner_grid,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="equal",
        interpolation="none"
    )

    # Set parameter values as tick labels
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels(b_values)

    ax.set_yticks(range(len(s_values)))
    ax.set_yticklabels(s_values)

    ax.set_xlabel("Benefit (b)")
    ax.set_ylabel("Signaling cost (s)")
    ax.set_title("Dominant Strategy Across Parameter Combinations")

    # Add grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(b_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(s_values), 1), minor=True)

    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colour legend
    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        ticks=np.arange(len(categories))
    )

    colorbar.ax.set_yticklabels(categories)
    colorbar.set_label("Winning strategy")

    plt.tight_layout()

    # Save plot
    save_dir = Path("results")
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / "strategy_winner_map.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Winner map saved to: {filepath}")

    plt.show()
    plt.close()


def main():

    if RUN_MODE == "test":

        run_single_test()

    elif RUN_MODE == "sweep":

        results, failed_simulations = run_parameter_sweep()

        plot_winner_map(results)

        for failure in failed_simulations:
            print(failure)

    else:

        raise ValueError("Unknown RUN_MODE.")


if __name__ == "__main__":
    main()