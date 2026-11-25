import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


from pathlib import Path
from datetime import datetime


def create_save_directory(params):

    parameter_folder = (
        f"b={params['b']:g}_"
        f"c={params['c']:g}_"
        f"s={params['s']:g}_"
        f"q={params['q']:g}"
    )

    rmech = params["rmech"]

    rmech_folder = "rmech_" + "_".join(
        "".join(str(value) for value in row)
        for row in rmech
    )

    save_dir = Path("Images") / parameter_folder / rmech_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir


def plot_strategy_dynamics(solution, params):

    t = solution.t
    names = list(params["strategies"].keys())
    n = len(names)

    shares = solution.y[:n]

    save_dir = create_save_directory(params)

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(save_dir / f"strategies_{timestamp}.png")
    plt.show()
    plt.close()

    print(f"Strategy plot saved to: {filepath}")


def plot_reputation_dynamics(solution, params):

    t = solution.t
    names = list(params["strategies"].keys())
    n = len(names)

    reputations = solution.y[n:]

    save_dir = create_save_directory(params)

    plt.figure(figsize=(10, 6))

    for i, name in enumerate(names):
        plt.plot(
            t,
            reputations[i],
            label=name,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(save_dir / f"reputations_{timestamp}.png")
    plt.show()
    plt.close()

    print(f"Reputation plot saved to: {filepath}")
