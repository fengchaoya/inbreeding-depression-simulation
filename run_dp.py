# run_dp.py
# Interface 3: Forward-looking (DP) Simulation
# Solves the Bellman equation via value function iteration

import numpy as np
import os

from core import GeneticOptimizer
from plotting import plot_dp_results, plot_bellman_objective

RESULTS_DIR = "results_dp"


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === Parameter Settings ===
    optimizer = GeneticOptimizer(
        h=0.6,
        Delta=0.4,
        beta=0.95,
        # Loss function parameters
        # Original dp_simulation.py: c * ((a*p)^2) * 2, i.e. c=1 => C1=2, C2=2
        C1=2.0,
        C2=2.0,
        n=2,
        m=2,
        cost_type='quadratic'
    )

    print("Starting dynamic optimization solver...")
    print(f"Parameters: h={optimizer.h}, Delta={optimizer.Delta}, "
          f"beta={optimizer.beta}, C1={optimizer.C1}, C2={optimizer.C2}")

    # Solve
    iterations = optimizer.value_function_iteration(max_iter=5000, tol=1e-6)

    # Steady-state analysis
    optimizer.analyze_steady_state()

    # Plot results
    plot_dp_results(optimizer, p0_list=[0.2, 0.5, 0.8], output_dir=RESULTS_DIR)

    print("\nGenerating Bellman equation objective function visualization...")
    plot_bellman_objective(optimizer, p_values_to_plot=[0.2, 0.5, 0.8],
                           output_dir=RESULTS_DIR)

    # Key policy values
    print("\nOptimal policy at key points:")
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for p in test_points:
        idx = min(int(p / optimizer.dp), optimizer.n_points - 1)
        print(f"  p = {p:.2f}: a* = {optimizer.policy[idx]:.3f}, "
              f"V = {optimizer.V[idx]:.3f}")
