import json
import time
import uuid
import numpy as np

from scipy.integrate import solve_ivp
from pathlib import Path
from datetime import datetime


"""
Model with a rudamentary update system
"""

def create_strategy(name, rmech):
    if name in ["ALLC", "ALLC_VS"]:
        return [0, 0, 0, 0]

    elif name in ["ALLD", "ALLD_VS"]:
        return [1, 1, 1, 1]

    elif name in ["DISC", "DISC_VS"]:
        return [
            0 if rmech[0][i] == 1 and rmech[1][i] == 0 else 1
            for i in range(4)
        ]

    else:
        raise ValueError(f"Unknown strategy: {name}")

def load_config(filepath="config.json"):

    with open(filepath, "r") as file:
        config = json.load(file)

    params = dict(config["parameters"])

    params["input_config"] = config

    rmech = config["reputation_mechanism"]

    strategies = {}
    signaling = {}

    for name, settings in config["strategies"].items():

        if not settings["enabled"]:
            continue

        strategies[name] = create_strategy(name, rmech)

        signaling[name] = name.endswith("_VS")

    params["rmech"] = rmech
    params["strategies"] = strategies
    params["signaling"] = signaling

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


def save_results(solution, params, runtime):

    names = list(params["strategies"].keys())
    n = len(names)

    simulation = params["simulation"]

    # Check convergence
    converged = len(solution.t_events[0]) > 0

    if converged:
        convergence_time = float(solution.t_events[0][-1])
        final_state = solution.y_events[0][-1]

        final_shares = shares_from_logits(final_state[:n-1])
        final_reputations = final_state[n-1:]

    else:
        convergence_time = None
        final_shares = solution.y[:n, -1]
        final_reputations = solution.y[n:, -1]

    # Determine outcome
    dominance_threshold = simulation["dominance_threshold"]

    winner_index = int(np.argmax(final_shares))

    if not converged:
        outcome = "not_converged"
        winner = None

    elif final_shares[winner_index] >= 1.0 - dominance_threshold:
        outcome = "dominance"
        winner = names[winner_index]

    else:
        outcome = "coexistence"
        winner = None

    # Build result dictionary
    results = {
        "simulation": {
            "converged": converged,
            "convergence_time": convergence_time,
            "runtime_seconds": float(runtime),
            "outcome": outcome,
            "winner": winner
        },

        "final_state": {
            "shares": {
                name: float(final_shares[i])
                for i, name in enumerate(names)
            },
            "reputations": {
                name: float(final_reputations[i])
                for i, name in enumerate(names)
            }
        },

        "input": params["input_config"]
    }

    # ========================================================
    # Create directory structure
    # ========================================================

    b = params["b"]
    c = params["c"]
    s = params["s"]
    q = params["q"]

    parameter_folder = (
        f"b={b:g}_c={c:g}_s={s:g}_q={q:g}"
    )

    # Encode reputation mechanism
    rmech = params["rmech"]

    rmech_folder = "rmech_" + "_".join(
        "".join(str(value) for value in row)
        for row in rmech
    )

    save_dir = (
        Path("results")
        / parameter_folder
        / rmech_folder
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Generate unique filename
    # ========================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:4]

    filename = f"simulation_{timestamp}_{unique_id}.json"

    filepath = save_dir / filename

    # Save JSON
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)

    print(f"\nConverged:          {converged}")
    print(f"Outcome:            {outcome}")
    print(f"Winner:             {winner if winner else 'None'}")
    print(f"Convergence time:   {convergence_time}")
    print(f"Runtime:            {runtime:.4f} seconds")

    print("\nFinal strategy shares:")

    for name, share in results["final_state"]["shares"].items():
        print(f"{name:<15} {share:.8f}")

    print("\n" + "=" * 50)
    print(f"Results saved to: {filepath}")

    return filepath



def normalize_shares(weights):

    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Initial weights must be non-negative.")

    total = np.sum(weights)

    if total <= 0:
        raise ValueError("The sum of initial weights must be positive.")

    return weights / total



def reputation_probabilities(g_i, g):

    return [
        g_i * g,
        g_i * (1 - g),
        (1 - g_i) * g,
        (1 - g_i) * (1 - g)
    ]


def update_reputation(g_i, g, rmech, strategy, signaling, q):

    probabilities = reputation_probabilities(g_i, g)

    assessments = [
        rmech[strategy[j]][j]
        for j in range(4)
    ]

    if signaling:
        assessments = [
            q if assessment == 0 else assessment
            for assessment in assessments
        ]

    return sum(
        probabilities[j] * assessments[j]
        for j in range(4)
    )


def cooperation_probability(strategy, g_i, g):

    probabilities = reputation_probabilities(g_i, g)

    return sum(
        probabilities[j] * (1 - strategy[j])
        for j in range(4)
    )


def payoff(shares, reputations, strategies, params):
    b = params["b"]
    c = params["c"]
    s = params["s"]

    signaling = params["signaling"]

    g = np.dot(shares, reputations)

    names = list(strategies.keys())
    payoffs = []

    for i, name_i in enumerate(names):

        benefit = 0.0

        for j, name_j in enumerate(names):
            benefit += shares[j] * b * cooperation_probability(
                strategies[name_j],
                reputations[j],
                reputations[i]
            )

        own_cooperation = cooperation_probability(
            strategies[name_i],
            reputations[i],
            g
        )

        cost = c * own_cooperation

        signaling_cost = s if signaling[name_i] else 0.0

        payoffs.append(benefit - cost - signaling_cost)

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
            strategies[name],
            params["signaling"][name],
            params["q"]
        ) - reputations[i]
        for i, name in enumerate(names)
    ])

    return np.concatenate((du, dg))



def convergence_event(t, current_state, params):

    n = len(params["strategies"])

    derivatives = replicator_eq(t, current_state, params)

    du = derivatives[:n-1]
    dg = derivatives[n-1:]

    # Reconstruct current shares
    u = current_state[:n-1]
    shares = shares_from_logits(u)

    # Calculate actual share derivatives from logit derivatives
    mean_relative_payoff = np.dot(shares[1:], du)

    dx = np.empty(n)

    dx[0] = -shares[0] * mean_relative_payoff

    dx[1:] = shares[1:] * (
        du - mean_relative_payoff
    )

    # Tolerances
    share_tolerance = params["simulation"]["share_tolerance"]
    reputation_tolerance = params["simulation"]["reputation_tolerance"]

    # Continuous convergence criterion
    share_error = np.max(np.abs(dx)) / share_tolerance
    reputation_error = np.max(np.abs(dg)) / reputation_tolerance

    return max(share_error, reputation_error) - 1.0


convergence_event.terminal = True
convergence_event.direction = -1



def solver(params):

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

    # Check normalization

    if not np.allclose(np.sum(shares, axis=0), 1.0, atol=1e-10):
        raise RuntimeError("Normalization condition violated!")

    if len(solution.t_events[0]) > 0:
        print("\nConvergence reached.")
    else:
        print("\nMaximum simulation time reached.")

    return solution

def run_simulation(config_path="config.json"):

    print("Loading configuration...")
    params = load_config(config_path)

    print("\nRunning simulation...")

    start_time = time.perf_counter()

    solution = solver(params)

    runtime = time.perf_counter() - start_time

    save_results(solution, params, runtime)

    return solution, params


if __name__ == "__main__":
    run_simulation()