import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
 
 
def create_save_directory(params):
 
    n_rounds = params["simulation"].get("n_rounds", 10)
 
    parameter_folder = (
        f"b={params['b']:g}_"
        f"c={params['c']:g}_"
        f"s={params['s']:g}_"
        f"q={params['q']:g}_"
        f"n={n_rounds}"
    )
 
    rmech = params["rmech"]
 
    rmech_folder = "rmech_" + "_".join(
        "".join(str(value) for value in row)
        for row in rmech
    )
 
    save_dir = Path("Images_generations") / parameter_folder / rmech_folder
    save_dir.mkdir(parents=True, exist_ok=True)
 
    return save_dir
 
 
def _finish_and_save(fig_name, save_dir, label):
 
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = save_dir / f"{fig_name}_{timestamp}.png"
 
    plt.savefig(filepath)
    plt.show()
    plt.close()
 
    print(f"{label} plot saved to: {filepath}")
 
 
def plot_strategy_dynamics(result, params):
    """Strategieanteile über die Generationen."""
 
    history = result["history"]
    names = result["names"]
 
    generations = history["generation"]
    shares = history["shares"]                  # Form: (n_generations + 1, n_strategies)
 
    save_dir = create_save_directory(params)
 
    plt.figure(figsize=(10, 6))
 
    for i, name in enumerate(names):
        plt.plot(generations, shares[:, i], label=name, linewidth=2)
 
    plt.xlabel("Generation")
    plt.ylabel("Strategy share")
    plt.title("Replicator Dynamics (generations)")
    plt.ylim(-0.02, 1.02)
 
    _finish_and_save("strategies", save_dir, "Strategy")
 
 
def plot_reputation_dynamics(result, params):
    """Reputationen am Ende jeder Generation (nach n Runden)."""
 
    history = result["history"]
    names = result["names"]
 
    generations = history["generation"]
    reputations = history["final_reputations"]  # Form: (n_generations + 1, n_strategies)
 
    save_dir = create_save_directory(params)
 
    plt.figure(figsize=(10, 6))
 
    for i, name in enumerate(names):
        plt.plot(generations, reputations[:, i], label=name, linewidth=2)
 
    plt.xlabel("Generation")
    plt.ylabel("Fraction with good reputation (end of generation)")
    plt.title("Reputation at end of generation")
    plt.ylim(-0.02, 1.02)
 
    _finish_and_save("reputations", save_dir, "Reputation")
 
 
def plot_round_reputations(result, params):
    """Reputationen Runde für Runde innerhalb der ersten und letzten Generation."""
 
    names = result["names"]
    trajectories = result["round_reputations"]
 
    save_dir = create_save_directory(params)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
 
    for ax, (key, title) in zip(
        axes, [("first", "First generation"), ("last", "Last generation")]
    ):
        traj = trajectories[key]                # Form: (n_rounds + 1, n_strategies)
 
        for i, name in enumerate(names):
            ax.plot(range(len(traj)), traj[:, i], label=name,
                    linewidth=2, marker="o", markersize=4)
 
        ax.set_xlabel("Round within generation")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
 
    axes[0].set_ylabel("Fraction with good reputation")
    axes[0].legend()
 
    plt.tight_layout()
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = save_dir / f"round_reputations_{timestamp}.png"
 
    plt.savefig(filepath)
    plt.show()
    plt.close()
 
    print(f"Round reputation plot saved to: {filepath}")
 