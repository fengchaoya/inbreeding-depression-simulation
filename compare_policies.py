# compare_policies.py
# Compare Myopic Regulator and Forward-looking (DP) Regulator policy functions

import numpy as np
import os

from core import calculate_policy_function, GeneticOptimizer
from plotting import plot_policy_comparison


def run_comparison():
    """Compute both policy functions and generate comparison plot."""
    print("--- Starting Myopic vs. Dynamic policy comparison ---")

    # === Shared parameters ===
    SHARED_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
    }

    # === Loss function parameters (unified across both models) ===
    LOSS_PARAMS = {
        "C1": 2.0,
        "C2": 2.0,
        "n": 2,
        "m": 2,
    }

    # === DP-specific parameters ===
    DP_PARAMS = {
        "beta": 0.95,
        "cost_type": 'quadratic',
    }

    p_grid = np.linspace(0, 1, 501)
    a_grid = np.linspace(0, 1, 501)

    # --- 1. Myopic Policy ---
    print("\n[1/2] Computing Myopic Policy Function...")
    myopic_params = {**SHARED_PARAMS, **LOSS_PARAMS}
    policy_myopic = calculate_policy_function(
        p_values=p_grid, ai_values=a_grid, static_params=myopic_params
    )

    # --- 2. Dynamic Policy ---
    print("\n[2/2] Computing Dynamic Policy Function...")
    optimizer = GeneticOptimizer(
        h=SHARED_PARAMS['h'],
        Delta=SHARED_PARAMS['Delta'],
        beta=DP_PARAMS['beta'],
        C1=LOSS_PARAMS['C1'],
        C2=LOSS_PARAMS['C2'],
        n=LOSS_PARAMS['n'],
        m=LOSS_PARAMS['m'],
        cost_type=DP_PARAMS['cost_type']
    )
    optimizer.value_function_iteration(verbose=False, tol=1e-6)

    # --- 3. Plot ---
    output_dir = "results_comparison"
    plot_policy_comparison(
        p_grid_myopic=p_grid,
        policy_myopic=policy_myopic,
        p_grid_dp=optimizer.p_grid,
        policy_dp=optimizer.policy,
        shared_params=SHARED_PARAMS,
        dp_params=DP_PARAMS,
        output_dir=output_dir
    )


if __name__ == '__main__':
    run_comparison()
