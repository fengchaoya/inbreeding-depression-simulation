# run_myopic.py
# Interface 2: Myopic (Single-Period) Regulator Simulation
# Features: policy function analysis, dynamic evolution simulation, yield curve visualization

import numpy as np
import os
import datetime

from core import calculate_policy_function, run_dynamic_policy_simulation
from plotting import (plot_policy_function, plot_p_evolution_comparison,
                      plot_yield_curves)


if __name__ == '__main__':
    # === Control Panel ===
    RUN_POLICY_FUNCTION_ANALYSIS = True
    RUN_DYNAMIC_SIMULATION = True
    RUN_YIELD_CURVE_EXPLANATION = True

    # === Parameter Settings ===
    RESULTS_DIR = "results_myopic"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    MODEL_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
        "population_size": 100000,
        "initial_A_proportion": 0.6,
        # Loss function parameters (set here for consistency)
        "C1": 2.0,
        "C2": 2.0,
        "n": 2,
        "m": 2,
    }

    p_values_grid = np.linspace(0, 1, 501)
    a_values_grid = np.linspace(0, 1, 501)

    # === Task 1: Policy Function a*(p) ===
    if RUN_POLICY_FUNCTION_ANALYSIS:
        print("\n" + "=" * 50)
        print("Task 1: Compute optimal policy function a*(p)")
        print("=" * 50)

        optimal_policy = calculate_policy_function(
            p_values=p_values_grid,
            ai_values=a_values_grid,
            static_params=MODEL_PARAMS
        )
        plot_policy_function(p_values_grid, optimal_policy,
                             MODEL_PARAMS, RESULTS_DIR)

    # === Task 2: Dynamic Evolution Simulation ===
    if RUN_DYNAMIC_SIMULATION:
        print("\n" + "=" * 50)
        print("Task 2: Run dynamic evolution from multiple initial conditions")
        print("=" * 50)

        initial_p_values_to_test = [0.2, 0.5, 0.8]
        total_generations = 10
        all_simulation_results = []

        for p0 in initial_p_values_to_test:
            print(f"  - Running simulation for p0 = {p0:.2f}...")
            current_sim_params = MODEL_PARAMS.copy()
            current_sim_params['initial_A_proportion'] = p0

            dynamic_results = run_dynamic_policy_simulation(
                total_generations=total_generations,
                ai_values=a_values_grid,
                static_params=current_sim_params
            )
            all_simulation_results.append(dynamic_results)

        print("\nAll simulations complete.")

        plot_p_evolution_comparison(
            results_list=all_simulation_results,
            initial_p_list=initial_p_values_to_test,
            static_params=MODEL_PARAMS,
            output_dir=RESULTS_DIR
        )

        # Save results
        for i, p0 in enumerate(initial_p_values_to_test):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"dynamic_sim_p0_{p0:.2f}_{timestamp}.npy"
            np.save(os.path.join(RESULTS_DIR, filename),
                    all_simulation_results[i])

    # === Task 3: Yield Curve Explanation ===
    if RUN_YIELD_CURVE_EXPLANATION:
        print("\n" + "=" * 50)
        print("Task 3: Generate yield curve explanation plots")
        print("=" * 50)

        p_values_for_explanation = [0.3, 0.5, 0.8]
        plot_yield_curves(
            p_values_to_show=p_values_for_explanation,
            ai_values=a_values_grid,
            static_params=MODEL_PARAMS,
            results_dir=RESULTS_DIR
        )

    print("\nAll selected tasks complete.")
