import json
import time
import uuid
import numpy as np
 
from pathlib import Path
from datetime import datetime
 
 
# ============================================================
# Strategien & Config
# ============================================================
 
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
    strategies, signaling = {}, {}
 
    for name, settings in config["strategies"].items():
        if not settings["enabled"]:
            continue
        strategies[name] = create_strategy(name, rmech)
        signaling[name] = name.endswith("_VS")
 
    params["rmech"] = rmech
    params["strategies"] = strategies
    params["signaling"] = signaling
    params["initial_shares"] = {n: config["strategies"][n]["share"] for n in strategies}
    params["initial_reputations"] = {n: config["strategies"][n]["reputation"] for n in strategies}
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
 
 
def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)
 
 
# ============================================================
# Reputation & Payoff
# ============================================================
 
def reputation_probabilities(g_i, g):
    return [
        g_i * g,
        g_i * (1 - g),
        (1 - g_i) * g,
        (1 - g_i) * (1 - g)
    ]
 
 
def update_reputation(g_i, g, rmech, strategy, signaling, q):
    """Rechte Seite F(g_i) der bisherigen Reputations-DGL: g_i' = F - g_i."""
    probabilities = reputation_probabilities(g_i, g)
    assessments = [rmech[strategy[j]][j] for j in range(4)]
 
    if signaling:
        assessments = [q if a == 0 else a for a in assessments]
 
    return sum(probabilities[j] * assessments[j] for j in range(4))
 
 
def cooperation_probability(strategy, g_i, g):
    probabilities = reputation_probabilities(g_i, g)
    return sum(probabilities[j] * (1 - strategy[j]) for j in range(4))
 
 
def payoff(shares, reputations, strategies, params):
    """Auszahlung pro Runde bei festen Anteilen und aktuellen Reputationen."""
    b, c, s = params["b"], params["c"], params["s"]
    signaling = params["signaling"]
 
    g = np.dot(shares, reputations)
    names = list(strategies.keys())
    payoffs = []
 
    for i, name_i in enumerate(names):
        benefit = sum(
            shares[j] * b * cooperation_probability(
                strategies[name_j], reputations[j], reputations[i]
            )
            for j, name_j in enumerate(names)
        )
        cost = c * cooperation_probability(strategies[name_i], reputations[i], g)
        signaling_cost = s if signaling[name_i] else 0.0
        payoffs.append(benefit - cost - signaling_cost)
 
    return g, np.array(payoffs)
 
 
def reputation_step(shares, reputations, params, dt):
    """
    Diskreter Reputationsschritt (Anteile sind fix).
 
    g_i(t+1) = g_i(t) + dt * (F_i - g_i(t))
    """
    names = list(params["strategies"].keys())
    g = np.dot(shares, reputations)
 
    new = np.empty_like(reputations)
    for i, name in enumerate(names):
        F = update_reputation(
            reputations[i], g, params["rmech"],
            params["strategies"][name], params["signaling"][name], params["q"]
        )
        new[i] = reputations[i] + dt * (F - reputations[i])
 
    return np.clip(new, 0.0, 1.0)
 
 
# ============================================================
# Eine Generation: n Runden (Anteile fix, nur Reputationen ändern sich)
# ============================================================
 
def run_generation(shares, reputations, params):
    sim = params["simulation"]
    n_rounds = sim.get("n_rounds", 10)
    dt = sim.get("reputation_step", 1.0)
 
    total_payoffs = np.zeros(len(shares))
    trajectory = [reputations.copy()]
 
    for _ in range(n_rounds):
        _, round_payoffs = payoff(shares, reputations, params["strategies"], params)
        total_payoffs += round_payoffs
        reputations = reputation_step(shares, reputations, params, dt)
        trajectory.append(reputations.copy())
 
    return reputations, total_payoffs, np.array(trajectory)
 
 
# ============================================================
# Klassifikation des aktuellen Zustands
# ============================================================
 
def assess_state(shares, payoffs, share_change, rep_change, sim):
    """
    shares, payoffs gehören zusammen: payoffs sind die mittleren Auszahlungen
    pro Runde, die bei diesen Anteilen erzielt wurden.
 
    Rückgabe (kind, w):
      "dominance"  : Anteil von w >= 1 - dominance_threshold UND kein anderer
                     hat eine höhere Auszahlung -> eindeutiger Gewinner
      "invadable"  : Anteil von w >= 1 - dominance_threshold, ABER eine andere
                     Strategie hat höhere Auszahlung -> kann invadieren
      "equilibrium": Anteile ändern sich nicht mehr, alle Strategien im Support
                     haben gleiche Auszahlung, keine fehlende Strategie kann
                     invadieren (-> Koexistenz)
      None         : noch nichts davon
    """
    n = len(shares)
    dom_thr = sim["dominance_threshold"]
    pay_tol = sim.get("payoff_tolerance", 1e-9)
    support_thr = sim.get("support_threshold", dom_thr)
 
    w = int(np.argmax(shares))
 
    if n == 1:
        return "dominance", w
 
    others = np.arange(n) != w
 
    # --- nahe an einer Ecke des Simplex -------------------------------------
    if np.sum(shares[others]) <= dom_thr:
        if np.max(payoffs[others]) <= payoffs[w] + pay_tol:
            return "dominance", w
        return "invadable", w
 
    # --- Gleichgewicht mit mehreren Strategien -------------------------------
    mean_payoff = float(np.dot(shares, payoffs))
    in_support = shares > support_thr
 
    equal_payoffs = np.all(np.abs(payoffs[in_support] - mean_payoff) <= pay_tol)
    no_invader = np.all(payoffs[~in_support] <= mean_payoff + pay_tol)
 
    if (
        equal_payoffs
        and no_invader
        and share_change < sim["share_tolerance"]
        and rep_change < sim["reputation_tolerance"]
    ):
        return "equilibrium", None
 
    return None, None
 
 
def log_depth(u, w):
    """
    ln(Gesamtanteil der anderen / Anteil des Monopolisten), berechnet aus den
    Logits. Funktioniert auch dann noch, wenn die Anteile in Floats unterlaufen.
    """
    others = np.delete(u, w)
    m = np.max(others)
    return float(m + np.log(np.sum(np.exp(others - m))) - u[w])
 
 
def cycle_detected(visits, min_visits):
    """
    Zyklus Richtung Rand: mindestens `min_visits` abgeschlossene Besuche, bei
    denen eine Strategie fast den ganzen Simplex einnimmt, aber invadiert werden
    kann; verschiedene Monopolisten wechseln sich ab, und jeder Besuch kommt
    näher an den Rand als der vorige (wachsende Oszillation).
    """
    if len(visits) < min_visits:
        return False
 
    last = visits[-min_visits:]
 
    if len({v["monopolist"] for v in last}) < 2:
        return False
 
    depths = [v["log_depth"] for v in last]
    return all(depths[i + 1] < depths[i] for i in range(len(depths) - 1))
 
 
# ============================================================
# Solver: Schleife über Generationen
# ============================================================
 
def solver(params):
    sim = params["simulation"]
    names = list(params["strategies"].keys())
    n = len(names)
 
    shares0 = normalize_shares([params["initial_shares"][nm] for nm in names])
    rep0 = np.array([params["initial_reputations"][nm] for nm in names], dtype=float)
 
    if np.any(shares0 <= 0):
        raise ValueError("Strategy shares must be positive")
    if np.any(rep0 < 0) or np.any(rep0 > 1):
        raise ValueError("Reputations must be between 0 and 1.")
 
    reset = sim.get("reset_reputations", True)
    n_generations = sim.get("n_generations", 100000)
    n_rounds = sim.get("n_rounds", 10)
    eta = sim.get("selection_strength", 0.1)
    min_visits = sim.get("cycle_min_visits", 3)
    support_thr = sim.get("support_threshold", sim["dominance_threshold"])
 
    # Wir halten die Logits statt der Anteile: so können Anteile beliebig nahe
    # an 0 kommen (und trotzdem wieder wachsen), ohne in Floats zu unterlaufen.
    u = np.log(shares0)
 
    reputations = rep0.copy()
    prev_final_reps = None
 
    history = {"generation": [], "shares": [], "final_reputations": [], "payoffs": []}
    visits = []
    current_visit = None
 
    outcome, winner, stop_generation = None, None, None
    first_trajectory = last_trajectory = None
 
    for gen in range(n_generations):
        shares = softmax(u)
 
        if reset:
            reputations = rep0.copy()
 
        reputations, total_payoffs, trajectory = run_generation(shares, reputations, params)
        payoffs = total_payoffs / n_rounds          # mittlere Auszahlung pro Runde
 
        if first_trajectory is None:
            first_trajectory = trajectory
        last_trajectory = trajectory
 
        # Replicator-Schritt (noch nicht angewendet, nur für die Änderung)
        u_next = u + eta * total_payoffs
        u_next -= np.max(u_next)
 
        share_change = float(np.max(np.abs(softmax(u_next) - shares)))
        rep_change = (
            float(np.max(np.abs(reputations - prev_final_reps)))
            if prev_final_reps is not None else np.inf
        )
        prev_final_reps = reputations.copy()
 
        history["generation"].append(gen)
        history["shares"].append(shares.copy())
        history["final_reputations"].append(reputations.copy())
        history["payoffs"].append(payoffs.copy())
 
        kind, w = assess_state(shares, payoffs, share_change, rep_change, sim)
 
        # --- Besuche "Monopolist, aber invadierbar" mitschreiben -------------
        visit_ended = False
 
        if kind == "invadable":
            depth = log_depth(u, w)
            if current_visit is not None and current_visit["monopolist"] == names[w]:
                current_visit["log_depth"] = min(current_visit["log_depth"], depth)
                current_visit["end_generation"] = gen
            else:
                if current_visit is not None:
                    visit_ended = True
                others = np.arange(n) != w
                invader = int(np.argmax(np.where(others, payoffs, -np.inf)))
                current_visit = {
                    "start_generation": gen,
                    "end_generation": gen,
                    "monopolist": names[w],
                    "invader": names[invader],
                    "log_depth": depth,
                }
                visits.append(current_visit)
        else:
            if current_visit is not None:
                visit_ended = True
            current_visit = None
 
        # --- Entscheidung -----------------------------------------------------
        if kind == "dominance":
            outcome, winner, stop_generation = "dominance", names[w], gen
            break
 
        if kind == "equilibrium":
            outcome, stop_generation = "coexistence", gen
            break
 
        if visit_ended:
            completed = visits[:-1] if current_visit is not None else visits
            if cycle_detected(completed, min_visits):
                outcome, stop_generation = "heteroclinic_cycle", gen
                break
 
        u = u_next
 
    # Maximale Generationenzahl erreicht
    if outcome is None:
        if cycle_detected(visits, min_visits):
            outcome, stop_generation = "heteroclinic_cycle", n_generations - 1
        else:
            outcome = "not_converged"
 
    final_shares = history["shares"][-1]
    coexisting = [names[i] for i in range(n) if final_shares[i] > support_thr]
 
    for k in history:
        history[k] = np.array(history[k])
 
    messages = {
        "dominance": "\nClear winner found.",
        "coexistence": "\nEquilibrium with multiple strategies reached.",
        "heteroclinic_cycle": "\nCycling toward the boundary of the simplex detected.",
        "not_converged": "\nMaximum number of generations reached without classification.",
    }
    print(messages[outcome])
 
    return {
        "outcome": outcome,
        "winner": winner,
        "coexisting_strategies": coexisting if outcome == "coexistence" else None,
        "convergence_generation": stop_generation if outcome != "not_converged" else None,
        "visits": visits,
        "history": history,
        "names": names,
        "round_reputations": {"first": first_trajectory, "last": last_trajectory},
    }
 
 
# ============================================================
# Ergebnisse speichern
# ============================================================
 
def save_results(result, params, runtime):
    names = result["names"]
    history = result["history"]
    sim = params["simulation"]
 
    outcome = result["outcome"]
    winner = result["winner"]
 
    final_shares = history["shares"][-1]
    final_reputations = history["final_reputations"][-1]
    final_payoffs = history["payoffs"][-1]
 
    results = {
        "simulation": {
            "outcome": outcome,
            "winner": winner,
            "coexisting_strategies": result["coexisting_strategies"],
            "convergence_generation": result["convergence_generation"],
            "n_generations_run": int(history["generation"][-1]) + 1,
            "runtime_seconds": float(runtime),
        },
        "final_state": {
            "shares": {n: float(final_shares[i]) for i, n in enumerate(names)},
            "mean_payoff_per_round": {n: float(final_payoffs[i]) for i, n in enumerate(names)},
            "reputations_end_of_generation": {
                n: float(final_reputations[i]) for i, n in enumerate(names)
            },
        },
        "cycle_visits": result["visits"],
        "input": params["input_config"],
    }
 
    b, c, s, q = params["b"], params["c"], params["s"], params["q"]
    parameter_folder = f"b={b:g}_c={c:g}_s={s:g}_q={q:g}_n={sim.get('n_rounds', 10)}"
    rmech_folder = "rmech_" + "_".join(
        "".join(str(v) for v in row) for row in params["rmech"]
    )
 
    save_dir = Path("results_generations") / parameter_folder / rmech_folder
    save_dir.mkdir(parents=True, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_dir / f"simulation_{timestamp}_{uuid.uuid4().hex[:4]}.json"
 
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)
 
    if sim.get("save_history", False):
        np.savez(
            filepath.with_suffix(".npz"),
            generation=history["generation"],
            shares=history["shares"],
            final_reputations=history["final_reputations"],
            payoffs=history["payoffs"],
            names=np.array(names),
        )
 
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"\nOutcome:                {outcome}")
    print(f"Winner:                 {winner if winner else 'None'}")
    if result["coexisting_strategies"]:
        print(f"Coexisting:             {', '.join(result['coexisting_strategies'])}")
    print(f"Stopped at generation:  {result['convergence_generation']}")
    print(f"Runtime:                {runtime:.4f} seconds")
 
    print("\nFinal strategy shares:")
    for name, share in results["final_state"]["shares"].items():
        print(f"{name:<15} {share:.8e}")
 
    if result["visits"]:
        print("\nVisits near the boundary (monopolist can be invaded):")
        for v in result["visits"]:
            print(
                f"  gen {v['start_generation']:>6}-{v['end_generation']:<6} "
                f"{v['monopolist']:<8} -> invader {v['invader']:<8} "
                f"log(depth) = {v['log_depth']:.2f}"
            )
 
    print("\n" + "=" * 50)
    print(f"Results saved to: {filepath}")
 
    return filepath
 
 
def run_simulation(config_path="config.json"):
    print("Loading configuration...")
    params = load_config(config_path)
 
    print("\nRunning simulation...")
    start = time.perf_counter()
    result = solver(params)
    runtime = time.perf_counter() - start
 
    save_results(result, params, runtime)
    return result, params
 
 
if __name__ == "__main__":
    run_simulation()