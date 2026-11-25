from model_generations import run_simulation
from plotting import (
    plot_strategy_dynamics,
    plot_reputation_dynamics,
    plot_round_reputations
)
 
def main():
 
    result, params = run_simulation()
 
    plot_strategy_dynamics(result, params)
    plot_reputation_dynamics(result, params)
    plot_round_reputations(result, params)
 
 
if __name__ == "__main__":
    main()