import json
import time
import uuid
import warnings
import numpy as np

from pathlib import Path
from datetime import datetime

# ============================================================
# Numba (optional). Ohne Numba läuft derselbe Code als reines Python,
# nur deutlich langsamer.
# ============================================================

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func

    warnings.warn(
        "Numba nicht gefunden: Simulation läuft im langsamen Python-Modus. "
        "Installation: pip install numba",
        stacklevel=2,
    )


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


def parse_config(config):
    """Baut die Parameter aus einem Config-Dictionary (statt aus einer Datei)."""
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
    params["model"] = build_model(params)
    return params


def load_config(filepath="config.json"):
    with open(filepath, "r") as file:
        config = json.load(file)
    return parse_config(config)


def normalize_shares(weights):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("Initial weights must be non-negative.")
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("The sum of initial weights must be positive.")
    return weights / total


def build_model(params):
    """
    Einmalige Vorberechnung aller Größen, die nur von Strategien und Config
    abhängen (nicht vom Zustand).

      coop        (n,4): 1 - Strategie, d.h. Kooperationsentscheidung je Fall
      assess      (n,4): Bewertung der Handlung je Fall (inkl. q bei Signaling)
      signal_cost (n,) : s für Signaling-Strategien, sonst 0
    """
    names = list(params["strategies"].keys())
    rmech = params["rmech"]
    q = params["q"]

    S = np.array([params["strategies"][nm] for nm in names], dtype=int)
    sig = np.array([params["signaling"][nm] for nm in names], dtype=bool)

    assess = np.array(
        [[rmech[S[i, j]][j] for j in range(4)] for i in range(len(names))],
        dtype=float
    )
    assess = np.where(sig[:, None] & (assess == 0), q, assess)

    return {
        "coop": np.ascontiguousarray(1.0 - S, dtype=np.float64),
        "assess": np.ascontiguousarray(assess, dtype=np.float64),
        "signal_cost": np.ascontiguousarray(params["s"] * sig, dtype=np.float64),
        "b": float(params["b"]),
        "c": float(params["c"]),
    }


# ============================================================
# Rechenkern (Numba): nur Arrays und Skalare
# ============================================================

@njit(cache=True)
def _softmax(u, out):
    n = u.shape[0]
    m = u[0]
    for i in range(1, n):
        if u[i] > m:
            m = u[i]
    total = 0.0
    for i in range(n):
        out[i] = np.exp(u[i] - m)
        total += out[i]
    for i in range(n):
        out[i] = out[i] / total


@njit(cache=True)
def _round(shares, r, coop, assess, signal_cost, b, c, dt, pay, new_r):
    """
    Eine Runde bei festen Anteilen: Auszahlungen pay (bei Reputationen r) und
    neue Reputationen new_r.

    Die vier Fälle (Donor-Reputation r_i, Population g) haben die
    Wahrscheinlichkeiten  U[i,k] * Vg[k]  mit
        U[i] = [r_i, r_i, 1-r_i, 1-r_i],   Vg = [g, 1-g, g, 1-g].
    """
    n = shares.shape[0]

    g = 0.0
    for i in range(n):
        g += shares[i] * r[i]

    # s[k] = sum_i shares[i] * U[i,k] * coop[i,k]
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for i in range(n):
        ri = r[i]
        s0 += shares[i] * ri * coop[i, 0]
        s1 += shares[i] * ri * coop[i, 1]
        s2 += shares[i] * (1.0 - ri) * coop[i, 2]
        s3 += shares[i] * (1.0 - ri) * coop[i, 3]

    for i in range(n):
        ri = r[i]
        benefit = b * (ri * s0 + (1.0 - ri) * s1 + ri * s2 + (1.0 - ri) * s3)
        own = (
            ri * coop[i, 0] * g
            + ri * coop[i, 1] * (1.0 - g)
            + (1.0 - ri) * coop[i, 2] * g
            + (1.0 - ri) * coop[i, 3] * (1.0 - g)
        )
        pay[i] = benefit - c * own - signal_cost[i]

        F = (
            ri * assess[i, 0] * g
            + ri * assess[i, 1] * (1.0 - g)
            + (1.0 - ri) * assess[i, 2] * g
            + (1.0 - ri) * assess[i, 3] * (1.0 - g)
        )
        v = ri + dt * (F - ri)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        new_r[i] = v


@njit(cache=True)
def _run_generation(shares, rep, coop, assess, signal_cost, b, c, dt,
                    n_rounds, fix_tol, total_pay, traj, record, pay, new_r):
    """
    n_rounds Runden, Anteile fix. rep wird in place auf die End-Reputationen
    gesetzt, total_pay enthält die Summe der Auszahlungen.

    Sind die Reputationen stabil (Änderung <= fix_tol), liefern alle weiteren
    Runden dieselbe Auszahlung: Die Restrunden werden mit dieser Auszahlung
    mitgezählt, die Reputationen aber nicht mehr neu bestimmt.
    """
    n = shares.shape[0]

    for i in range(n):
        total_pay[i] = 0.0
    if record:
        for i in range(n):
            traj[0, i] = rep[i]

    for k in range(n_rounds):
        _round(shares, rep, coop, assess, signal_cost, b, c, dt, pay, new_r)

        stable = True
        for i in range(n):
            total_pay[i] += pay[i]
            if abs(new_r[i] - rep[i]) > fix_tol:
                stable = False
            rep[i] = new_r[i]

        if record:
            for i in range(n):
                traj[k + 1, i] = rep[i]

        if stable:
            remaining = n_rounds - k - 1
            if remaining > 0:
                _round(shares, rep, coop, assess, signal_cost, b, c, dt, pay, new_r)
                for i in range(n):
                    total_pay[i] += remaining * pay[i]
                if record:
                    for j in range(k + 2, n_rounds + 1):
                        for i in range(n):
                            traj[j, i] = rep[i]
            break


@njit(cache=True)
def _assess(shares, pay, share_change, rep_change,
            dom_thr, pay_tol, support_thr, share_tol, rep_tol):
    """
    Rückgabe (code, w):
      1 = eindeutiger Gewinner w (Anteil >= 1 - dom_thr und keine andere
          Strategie hat höhere Auszahlung)
      2 = Gleichgewicht mit mehreren Strategien (Anteile/Reputationen stabil,
          gleiche Auszahlung im Support, keine fehlende Strategie kann invadieren)
      0 = noch nichts davon
    """
    n = shares.shape[0]

    w = 0
    for i in range(1, n):
        if shares[i] > shares[w]:
            w = i

    if n == 1:
        return 1, w

    sum_others = 0.0
    max_other_pay = -np.inf
    for i in range(n):
        if i != w:
            sum_others += shares[i]
            if pay[i] > max_other_pay:
                max_other_pay = pay[i]

    # nahe an einer Ecke des Simplex
    if sum_others <= dom_thr:
        if max_other_pay <= pay[w] + pay_tol:
            return 1, w
        return 0, w

    # Gleichgewicht mit mehreren Strategien
    mean_pay = 0.0
    for i in range(n):
        mean_pay += shares[i] * pay[i]

    equal_payoffs = True
    no_invader = True
    for i in range(n):
        if shares[i] > support_thr:
            if abs(pay[i] - mean_pay) > pay_tol:
                equal_payoffs = False
        else:
            if pay[i] > mean_pay + pay_tol:
                no_invader = False

    if (
        equal_payoffs
        and no_invader
        and share_change < share_tol
        and rep_change < rep_tol
    ):
        return 2, w

    return 0, w


@njit(cache=True)
def _simulate(shares0, rep0, coop, assess, signal_cost, b, c, dt,
              n_rounds, n_generations, eta, reset, fix_tol,
              dom_thr, pay_tol, support_thr, share_tol, rep_tol, record):
    """
    Generationenschleife. Rückgabe:
      (outcome_code, winner, stop_generation, n_run,
       hist_shares, hist_rep, hist_pay, traj_first, traj_last,
       last_shares, last_rep, last_pay)
    outcome_code: 0 nicht konvergiert, 1 Gewinner, 2 Koexistenz.
    Ohne record sind die hist_/traj_-Arrays nur Platzhalter (1 Zeile).
    """
    n = shares0.shape[0]

    n_hist = n_generations if record else 1
    n_traj = n_rounds + 1 if record else 1

    hist_shares = np.zeros((n_hist, n))
    hist_rep = np.zeros((n_hist, n))
    hist_pay = np.zeros((n_hist, n))
    traj = np.zeros((n_traj, n))
    traj_first = np.zeros((n_traj, n))

    u = np.log(shares0)
    u_next = np.zeros(n)
    shares = np.zeros(n)
    shares_next = np.zeros(n)
    rep = rep0.copy()
    prev_rep = np.zeros(n)
    total_pay = np.zeros(n)
    pay = np.zeros(n)
    new_r = np.zeros(n)
    mean_pay = np.zeros(n)

    outcome = 0
    winner = -1
    stop_generation = -1
    n_run = 0

    for gen in range(n_generations):
        _softmax(u, shares)

        if reset:
            for i in range(n):
                rep[i] = rep0[i]

        _run_generation(shares, rep, coop, assess, signal_cost, b, c, dt,
                        n_rounds, fix_tol, total_pay, traj, record, pay, new_r)

        for i in range(n):
            mean_pay[i] = total_pay[i] / n_rounds

        # Replicator-Schritt (nur vorbereiten; angewendet wird er erst unten).
        # Es geht die MITTLERE Auszahlung pro Runde ein, dadurch ist die
        # Selektionsstärke eta unabhängig von n_rounds.
        m = -np.inf
        for i in range(n):
            u_next[i] = u[i] + eta * mean_pay[i]
            if u_next[i] > m:
                m = u_next[i]
        for i in range(n):
            u_next[i] -= m
        _softmax(u_next, shares_next)

        share_change = 0.0
        for i in range(n):
            d = abs(shares_next[i] - shares[i])
            if d > share_change:
                share_change = d

        if gen == 0:
            rep_change = np.inf
        else:
            rep_change = 0.0
            for i in range(n):
                d = abs(rep[i] - prev_rep[i])
                if d > rep_change:
                    rep_change = d
        for i in range(n):
            prev_rep[i] = rep[i]

        if record:
            for i in range(n):
                hist_shares[gen, i] = shares[i]
                hist_rep[gen, i] = rep[i]
                hist_pay[gen, i] = mean_pay[i]
            if gen == 0:
                for j in range(n_traj):
                    for i in range(n):
                        traj_first[j, i] = traj[j, i]

        n_run = gen + 1

        code, w = _assess(shares, mean_pay, share_change, rep_change,
                          dom_thr, pay_tol, support_thr, share_tol, rep_tol)

        if code == 1:
            outcome = 1
            winner = w
            stop_generation = gen
            break
        if code == 2:
            outcome = 2
            stop_generation = gen
            break

        for i in range(n):
            u[i] = u_next[i]

    return (outcome, winner, stop_generation, n_run,
            hist_shares, hist_rep, hist_pay, traj_first, traj,
            shares.copy(), rep.copy(), mean_pay.copy())


def warm_up():
    """Kompiliert (bzw. lädt aus dem Cache) den Numba-Kern mit einer Mini-Simulation."""
    n = 2
    _simulate(
        np.array([0.5, 0.5]), np.array([0.5, 0.5]),
        np.ones((n, 4)), np.ones((n, 4)), np.zeros(n),
        1.0, 0.5, 1.0, 2, 2, 0.1, True, 1e-12,
        1e-6, 1e-9, 1e-6, 1e-9, 1e-9, False,
    )


# ============================================================
# Solver (Python-Hülle um den Rechenkern)
# ============================================================

OUTCOMES = {0: "not_converged", 1: "dominance", 2: "coexistence"}

MESSAGES = {
    "dominance": "\nClear winner found.",
    "coexistence": "\nEquilibrium with multiple strategies reached.",
    "not_converged": "\nMaximum number of generations reached without classification.",
}


def solver(params, record_history=True, verbose=True):
    sim = params["simulation"]
    names = list(params["strategies"].keys())
    n = len(names)

    shares0 = normalize_shares([params["initial_shares"][nm] for nm in names])
    rep0 = np.array([params["initial_reputations"][nm] for nm in names], dtype=float)

    if np.any(shares0 <= 0):
        raise ValueError("Strategy shares must be positive")
    if np.any(rep0 < 0) or np.any(rep0 > 1):
        raise ValueError("Reputations must be between 0 and 1.")

    model = params["model"]
    dom_thr = float(sim["dominance_threshold"])

    (outcome_code, winner_idx, stop_generation, n_run,
     hist_shares, hist_rep, hist_pay, traj_first, traj_last,
     last_shares, last_rep, last_pay) = _simulate(
        shares0, rep0,
        model["coop"], model["assess"], model["signal_cost"],
        model["b"], model["c"],
        float(sim.get("reputation_step", 1.0)),
        int(sim.get("n_rounds", 10)),
        int(sim.get("n_generations", 100000)),
        float(sim.get("selection_strength", 0.1)),
        bool(sim.get("reset_reputations", True)),
        float(sim.get("reputation_fixpoint_tolerance", 1e-12)),
        dom_thr,
        float(sim.get("payoff_tolerance", 1e-9)),
        float(sim.get("support_threshold", dom_thr)),
        float(sim["share_tolerance"]),
        float(sim["reputation_tolerance"]),
        bool(record_history),
    )

    outcome = OUTCOMES[outcome_code]
    winner = names[winner_idx] if outcome == "dominance" else None

    support_thr = sim.get("support_threshold", dom_thr)
    coexisting = (
        [names[i] for i in range(n) if last_shares[i] > support_thr]
        if outcome == "coexistence" else None
    )

    if record_history:
        history = {
            "generation": np.arange(n_run),
            "shares": hist_shares[:n_run],
            "final_reputations": hist_rep[:n_run],
            "payoffs": hist_pay[:n_run],
        }
        round_reputations = {"first": traj_first, "last": traj_last}
    else:
        # nur der letzte Zustand (spart Speicher bei Sweeps)
        history = {
            "generation": np.array([n_run - 1]),
            "shares": last_shares[None, :],
            "final_reputations": last_rep[None, :],
            "payoffs": last_pay[None, :],
        }
        round_reputations = None

    if verbose:
        print(MESSAGES[outcome])

    return {
        "outcome": outcome,
        "winner": winner,
        "coexisting_strategies": coexisting,
        "convergence_generation": None if outcome == "not_converged" else int(stop_generation),
        "history": history,
        "names": names,
        "round_reputations": round_reputations,
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
            "runtime_seconds": float(runtime),
        },
        "final_state": {
            "shares": {n: float(final_shares[i]) for i, n in enumerate(names)},
            "mean_payoff_per_round": {n: float(final_payoffs[i]) for i, n in enumerate(names)},
            "reputations_end_of_generation": {
                n: float(final_reputations[i]) for i, n in enumerate(names)
            },
        },
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

    print("\n" + "=" * 50)
    print(f"Results saved to: {filepath}")

    return filepath


def run_simulation(config_path="config.json", config=None,
                   record_history=True, verbose=True):
    """
    config      : optional Config-Dictionary (statt Datei), z. B. für Sweeps
    record_history=False: speichert nur den letzten Zustand (schnell, wenig RAM)
    """
    if verbose:
        print("Loading configuration...")
    params = parse_config(config) if config is not None else load_config(config_path)

    if verbose:
        print("\nRunning simulation...")
    start_time = time.perf_counter()

    result = solver(params, record_history=record_history, verbose=verbose)

    result["runtime_seconds"] = float(time.perf_counter() - start_time)

    return result, params


if __name__ == "__main__":
    run_simulation()