import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm


def create_strategy(name, rmech):
    if name == "ALLC":
        return [0, 0, 0, 0]

    elif name == "ALLD":
        return [1, 1, 1, 1]

    elif name == "DISC":
        return [
            0 if rmech[0][i] == 1 and rmech[1][i] == 0 else 1
            for i in range(4)
        ]

    else:
        raise ValueError(f"Unknown strategy: {name}")

def load_config(filepath="config.json"):
    with open(filepath, "r") as file:
        config = json.load(file)

    params = config["parameters"]
    rmech = config["reputation_mechanism"]

    strategies = {}

    for name, settings in config["strategies"].items():

        if not settings["enabled"]:
            continue

        strategies[name] = create_strategy(name, rmech)

    params["rmech"] = rmech
    params["strategies"] = strategies

    params["initial_shares"] = {
        name: config["strategies"][name]["share"]
        for name in strategies
    }

    params["initial_reputations"] = {
        name: config["strategies"][name]["reputation"]
        for name in strategies
    }

    params["simulation"] = config["simulation"]

    return params


def normalize_shares(weights):

    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Initial weights must be non-negative.")

    total = np.sum(weights)

    if total <= 0:
        raise ValueError("The sum of initial weights must be positive.")

    return weights / total



def update_reputation(g_i, g, rmech, strategy):

    # Columns: GG, GB, BG, BB
    probabilities = [
        g_i * g,
        g_i * (1 - g),
        (1 - g_i) * g,
        (1 - g_i) * (1 - g)
    ]

    new_g = 0.0

    for j in range(4):

        action = strategy[j]

        # 0 = cooperation, 1 = defection
        assessment = rmech[action][j]

        new_g += probabilities[j] * assessment

    return new_g



def cooperation_probability(strategy, g_donor, g_recipient):

    probabilities = [
        g_donor * g_recipient,
        g_donor * (1 - g_recipient),
        (1 - g_donor) * g_recipient,
        (1 - g_donor) * (1 - g_recipient)
    ]

    return sum(
        probabilities[j] * (1 - strategy[j])
        for j in range(4)
    )


def payoff(shares, reputations, strategies, params):

    b = params["b"]
    c = params["c"]

    names = list(strategies.keys())

    # Total fraction with good reputation
    g = np.dot(shares, reputations)

    payoffs = []

    for i, name_i in enumerate(names):

        # Expected benefit received from the population
        benefit = 0.0

        for j, name_j in enumerate(names):

            benefit += (
                shares[j] * b *
                cooperation_probability(
                    strategies[name_j],
                    reputations[j],
                    reputations[i]
                )
            )

        # Expected cooperation by strategy i
        own_cooperation = cooperation_probability(
            strategies[name_i],
            reputations[i],
            g
        )

        cost = c * own_cooperation

        payoffs.append(benefit - cost)

    return g, np.array(payoffs)



def shares_from_logits(u):

    logits = np.concatenate(([0.0], u))

    logits -= np.max(logits)

    exp_u = np.exp(logits)

    return exp_u / np.sum(exp_u)


def replicator_eq(t, current_state, params):

    strategies = params["strategies"]
    rmech = params["rmech"]

    names = list(strategies.keys())

    n = len(names)

    # Split strategy and reputation variables

    u = current_state[:n-1]
    reputations = current_state[n-1:]

    # Reconstruct strategy shares

    shares = shares_from_logits(u)

    # Calculate payoffs

    g, payoffs = payoff(
        shares,
        reputations,
        strategies,
        params
    )

    # Replicator dynamics

    du = payoffs[1:] - payoffs[0]

    # Reputation dynamics

    dg = np.array([
        update_reputation(
            reputations[i],
            g,
            rmech,
            strategies[name]
        ) - reputations[i]

        for i, name in enumerate(names)
    ])

    return np.concatenate((du, dg))



def convergence_event(t, current_state, params):

    n = len(params["strategies"])

    u = current_state[:n-1]

    shares = shares_from_logits(u)

    return np.max(shares) - (1.0 - 1e-12)


convergence_event.terminal = True
convergence_event.direction = 1




def run_simulation(params):

    simulation = params["simulation"]

    t_start = simulation["t_start"]
    t_end = simulation["t_end"]
    t_points = simulation["t_points"]
    strategies = params["strategies"]

    names = list(strategies.keys())

    n = len(names)

    # ========================================================
    # Initial strategy shares
    # ========================================================

    initial_weights = np.array([
        params["initial_shares"][name]
        for name in names
    ])

    # Automatically normalize weights

    initial_shares = normalize_shares(initial_weights)

    # Initial reputations

    initial_reputations = np.array([
        params["initial_reputations"][name]
        for name in names
    ])

    # Transform initial shares into log-ratio coordinates

    u0 = np.log(
        initial_shares[1:] / initial_shares[0]
    )


    # Initial reputations

    initial_reputations = np.array([
        params["initial_reputations"][name]
        for name in names
    ])

    # Validate initial conditions

    if np.any(initial_shares <= 0):
        raise ValueError("Strategy shares must be positive")

    if np.any(initial_reputations < 0) or np.any(initial_reputations > 1):
        raise ValueError("Reputations must be between 0 and 1.")

    # Transform strategy shares into log-ratio coordinates

    u0 = np.log(initial_shares[1:] / initial_shares[0])

    # Complete initial state

    y0 = np.concatenate((u0, initial_reputations))

    t_eval = np.linspace(t_start, t_end, t_points)

    solution = solve_ivp(
        fun=replicator_eq,
        t_span=(t_start, t_end),
        y0=y0,
        t_eval=t_eval,
        args=(params,),
        method="DOP853",
        rtol=simulation["rtol"],
        atol=simulation["atol"],
        events=[convergence_event]
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # Reconstruct strategy shares

    u = solution.y[:n-1]

    shares = np.array([
        shares_from_logits(u[:, i])
        for i in range(len(solution.t))
    ]).T

    # Extract reputations

    reputations = solution.y[n-1:]

    # Store all variables

    solution.y = np.vstack((shares, reputations))

    # Print final results

    print("\nFinal strategy shares:")

    for i, name in enumerate(names):
        print(f"{name}(T) = {shares[i, -1]:.6f}")

    print("\nFinal reputations:")

    for i, name in enumerate(names):
        print(f"g_{name}(T) = {reputations[i, -1]:.6f}")

    # Check normalization

    if not np.allclose(np.sum(shares, axis=0), 1.0, atol=1e-10):
        raise RuntimeError("Normalization condition violated!")

    if len(solution.t_events[0]) > 0:
        print("\nConvergence reached.")
    else:
        print("\nMaximum simulation time reached.")

    return solution


def plot_solution(solution, params):

    t = solution.t

    names = list(params["strategies"].keys())

    n = len(names)

    # Extract variables

    shares = solution.y[:n]
    reputations = solution.y[n:]

    # Create parameter-specific directory

    parameter_folder = (
        f"b={params['b']:g}_"
        f"s={params['s']:g}_"
        f"q={params['q']:g}"
    )

    save_dir = Path("results") / parameter_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Plot 1: Strategy dynamics
    # ========================================================

    plt.figure(figsize=(10, 6))

    for i, name in enumerate(names):

        plt.plot(
            t,
            shares[i],
            label=name,
            linewidth=2
        )

    plt.xlabel("Time")
    plt.ylabel("Strategy share")
    plt.title("Replicator Dynamics")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filepath = save_dir / "strategies.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # ========================================================
    # Plot 2: Reputation dynamics
    # ========================================================

    plt.figure(figsize=(10, 6))

    for i, name in enumerate(names):

        plt.plot(
            t,
            reputations[i],
            label=f"$g_{{{name}}}$",
            linewidth=2
        )

    plt.xlabel("Time")
    plt.ylabel("Fraction with good reputation")
    plt.title("Reputation Dynamics")

    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filepath = save_dir / "reputations.png"

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"\nPlots saved to: {save_dir}")

def run_single_test():

    print("Loading configuration...")

    params = load_config("config.json")

    print("\nRunning simulation...")

    solution = run_simulation(params)

    plot_solution(solution, params)


if __name__ == "__main__":
    run_single_test()