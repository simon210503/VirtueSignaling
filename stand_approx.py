import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm


from itertools import product

# ============================================================
# Simulation settings
# ============================================================

RUN_MODE = "sweep"  # Options: "test" or "sweep"

BASE_PARAMS = {
    "b": 2.0,
    "c": 1.0,
    "s": 0.4,
    "g_x": 1,
    "g_y": 0,
    "g_z": 1,
    "g_v": 0.5,
    "q": 0.5
}

# Initial conditions for testing
TEST_INITIAL = (0.98889998, 0.01)


# ============================================================
# Parameter sweep settings
# ============================================================

PARAMETER_GRID = {
    "b": [2.0],
    "s": [0.1, 0.3, 0.5, 0.7, 0.9],
    "q": [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Different initial conditions
INITIAL_CONDITIONS = [
    (1/3, 1/3),
    (0.8, 0.1),
    (0.1, 0.8),
    (0.1, 0.1)
]


def payoff(y, z, v, params):

    b = params["b"]
    c = params["c"]
    s = params["s"]
    q = params["q"]

    g_y = params["g_y"]
    g_z = params["g_z"]
    g_v = q

    g = y * g_y + z * g_z + v * g_v

    P_y = b * (z * g_y)
    P_z = b * (z * g_z) - c * g
    P_v = b * (z * g_v) - s

    P_bar = y * P_y + z * P_z + v * P_v

    return g, P_y, P_z, P_v, P_bar


def replicator_eq(t, current_state, params):

    u_y, u_z = current_state

    # Reconstruct strategy shares using softmax
    max_u = max(0.0, u_y, u_z)

    exp_y = np.exp(u_y - max_u)
    exp_z = np.exp(u_z - max_u)
    exp_v = np.exp(-max_u)

    denominator = exp_y + exp_z + exp_v

    y = exp_y / denominator
    z = exp_z / denominator
    v = exp_v / denominator

    # Calculate payoffs
    g, P_y, P_z, P_v, P_bar = payoff(
        y, z, v, params
    )

    # Log-ratio replicator equations
    du_y = P_y - P_v
    du_z = P_z - P_v

    return [du_y, du_z]


def convergence_event(t, current_state, params):

    u_y, u_z = current_state

    max_u = max(0.0, u_y, u_z)

    exp_y = np.exp(u_y - max_u)
    exp_z = np.exp(u_z - max_u)
    exp_v = np.exp(-max_u)

    denominator = exp_y + exp_z + exp_v

    y = exp_y / denominator
    z = exp_z / denominator
    v = exp_v / denominator

    return max(y, z, v) - (1.0 - 1e-12)


convergence_event.terminal = True
convergence_event.direction = 1

def run_simulation(params, initial_conditions):

    t_start = 0.0
    t_end = 1000.0
    t_points = 15000

    # Initial strategy shares
    y0, z0 = initial_conditions
    v0 = 1.0 - y0 - z0

    initial_shares = np.array([y0, z0, v0])

    if np.any(initial_shares <= 0) or np.any(initial_shares > 1):
        raise ValueError(
            "All initial strategy shares must be strictly positive and <= 1."
        )

    if not np.isclose(np.sum(initial_shares), 1.0):
        raise ValueError("Initial strategy shares must sum to 1.")

    # Transform initial conditions into log-ratio coordinates
    y0_vector = np.array([
        np.log(y0 / v0),
        np.log(z0 / v0)
    ])

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

    if not solution.success:
        raise RuntimeError(solution.message)

    # Reconstruct strategy shares
    u_y, u_z = solution.y

    max_u = np.maximum.reduce([
        np.zeros_like(u_y),
        u_y,
        u_z
    ])

    exp_y = np.exp(u_y - max_u)
    exp_z = np.exp(u_z - max_u)
    exp_v = np.exp(-max_u)

    denominator = exp_y + exp_z + exp_v

    y = exp_y / denominator
    z = exp_z / denominator
    v = exp_v / denominator

    # Replace logarithmic solution with strategy shares
    solution.y = np.vstack([y, z, v])

    # Print final strategy shares
    print("\nFinal strategy shares:")
    print(f"y(T) = {y[-1]:.6f}")
    print(f"z(T) = {z[-1]:.6f}")
    print(f"v(T) = {v[-1]:.6f}")

    # Check normalization
    total = y + z + v

    if not np.allclose(total, 1.0, atol=1e-10):
        raise RuntimeError("Normalization condition violated!")

    if len(solution.t_events[0]) > 0:
        print("Convergence reached.")
    else:
        print("Maximum simulation time reached.")

    return solution

def plot_solution(solution, params, initial_conditions):

    t = solution.t
    y, z, v = solution.y

    # Create parameter-specific directory
    parameter_folder = f"b={params['b']:g}_s={params['s']:g}_q={params['q']:g}"

    save_dir = Path("results") / parameter_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initial conditions
    y0, z0 = initial_conditions
    v0 = 1 - y0 - z0

    filename = f"IC_y={y0:g}_z={z0:g}_v={v0:g}.png"

    filepath = save_dir / filename

    # Create plot
    plt.figure(figsize=(10, 6))

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

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {filepath}")

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

    # Store only aggregated winner counts
    winner_counts = {}

    failed_simulations = []

    strategy_names = [
        "ALLD",
        "Discriminator",
        "Pure Virtue Signaling"
    ]

    for values in product(*parameter_values):

        params = BASE_PARAMS.copy()

        for name, value in zip(parameter_names, values):
            params[name] = value

        for initial in INITIAL_CONDITIONS:

            try:
                solution = run_simulation(params, initial)

                # Plot trajectory if required
                plot_solution(solution, params, initial)

                # Extract only final shares
                final_shares = solution.y[:, -1]

                # Check convergence
                converged = len(solution.t_events[0]) > 0

                if converged:
                    winner = strategy_names[
                        np.argmax(final_shares)
                    ]
                else:
                    winner = "No winner"

                # Aggregate winner counts immediately
                key = (
                    params["b"],
                    params["s"],
                    params["q"]
                )

                if key not in winner_counts:
                    winner_counts[key] = {
                        strategy: 0
                        for strategy in strategy_names + ["No winner"]
                    }

                winner_counts[key][winner] += 1

                print(
                    f"b={params['b']}, "
                    f"c={params['c']}, "
                    f"s={params['s']}, "
                    f"q={params['q']} | "
                    f"Initial={initial} | "
                    f"Final={np.round(final_shares, 3).tolist()} | "
                    f"Winner={winner}"
                )

                # Release simulation memory
                del solution
                del final_shares

            except (RuntimeError, ValueError) as error:

                print("\nSimulation failed. Skipping...")
                print(
                    f"b={params['b']}, "
                    f"c={params['c']}, "
                    f"s={params['s']}, "
                    f"q={params['q']} | "
                    f"Initial={initial}"
                )
                print(f"Error: {error}")

                failed_simulations.append({
                    "params": params.copy(),
                    "initial": initial,
                    "error": str(error)
                })

                continue

    successful_simulations = sum(
        sum(counts.values())
        for counts in winner_counts.values()
    )

    print("\nParameter sweep completed.")
    print(f"Successful simulations: {successful_simulations}")
    print(f"Failed simulations: {len(failed_simulations)}")

    return winner_counts, failed_simulations



def plot_winner_map(winner_counts):

    b_values = PARAMETER_GRID["b"]
    s_values = PARAMETER_GRID["s"]
    q_values = PARAMETER_GRID["q"]

    strategy_names = [
        "ALLD",
        "Discriminator",
        "Pure Virtue Signaling"
    ]

    categories = strategy_names + ["No winner"]

    colors = [
        "tab:red",
        "tab:green",
        "tab:purple",
        "lightgray"
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(categories) + 0.5, 1),
        cmap.N
    )

    save_dir = Path("results") / "winner_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    for b in b_values:

        winner_grid = np.full(
            (len(s_values), len(q_values)),
            len(categories) - 1
        )

        for i, s in enumerate(s_values):
            for j, q in enumerate(q_values):

                counts = winner_counts.get((b, s, q))

                if counts is None:
                    continue

                winner_counts_per_strategy = {
                    strategy: counts[strategy]
                    for strategy in strategy_names
                }

                no_winner_count = counts["No winner"]

                max_wins = max(
                    winner_counts_per_strategy.values()
                )

                winners = [
                    strategy
                    for strategy, count
                    in winner_counts_per_strategy.items()
                    if count == max_wins
                ]

                if (
                    len(winners) == 1
                    and max_wins > no_winner_count
                ):
                    winner_grid[i, j] = strategy_names.index(
                        winners[0]
                    )

        fig, ax = plt.subplots(figsize=(9, 7))

        im = ax.imshow(
            winner_grid,
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="equal",
            interpolation="none"
        )

        ax.set_xticks(range(len(q_values)))
        ax.set_xticklabels(q_values)

        ax.set_yticks(range(len(s_values)))
        ax.set_yticklabels(s_values)

        ax.set_xlabel("Success rate (q)")
        ax.set_ylabel("Signaling cost (s)")
        ax.set_title(f"Winning Strategy (b={b:g})")

        ax.set_xticks(
            np.arange(-0.5, len(q_values), 1),
            minor=True
        )

        ax.set_yticks(
            np.arange(-0.5, len(s_values), 1),
            minor=True
        )

        ax.grid(
            which="minor",
            color="black",
            linewidth=0.5
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            ticks=range(len(categories))
        )

        cbar.ax.set_yticklabels(categories)

        plt.tight_layout()

        filepath = save_dir / f"winner_map_b={b:g}.png"

        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Winner map saved to: {filepath}")

        plt.close(fig)

        del winner_grid


def main():

    if RUN_MODE == "test":

        run_single_test()

    elif RUN_MODE == "sweep":

        winner_counts, failed_simulations = run_parameter_sweep()

        plot_winner_map(winner_counts)

        for failure in failed_simulations:
            print(failure)

    else:

        raise ValueError("Unknown RUN_MODE.")

if __name__ == "__main__":
    main()