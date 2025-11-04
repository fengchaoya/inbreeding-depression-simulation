# compare_policies.py
# A script to compare the policy functions of the Myopic Regulator and the Dynamic Regulator.

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Import core functionalities from your existing scripts ---
# Import the policy calculation function from the myopic script
from myopic_model_suite import calculate_policy_function
# Import the optimizer class from the dp script
from dp_simulation import GeneticOptimizer

# --- Optional: Code to resolve issues with displaying Chinese characters ---
# This can be removed or kept depending on your preference.
# For consistency, it's better to manage fonts in a central place if needed,
# but for now, we'll keep it as is.
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_comparison():
    """
    Runs the calculations for both policies and plots the comparison.
    """
    print("--- Starting comparison between Myopic and Dynamic policies ---")

    # --- 2. Define shared model parameters ---
    # Set parameters uniformly here to ensure both models are compared
    # under the same conditions.
    SHARED_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
    }

    # Parameters specific to the Myopic Regulator (pi, gamma)
    MYOPIC_EXTRA_PARAMS = {
        "pi": 0.5,
        "gamma": 1.0
    }

    # Parameters specific to the Dynamic Regulator (beta, c)
    DP_EXTRA_PARAMS = {
        "beta": 0.95,
        "c": 1.0,
        "cost_type": 'quadratic'
    }

    # Define the grid of p-values for the analysis
    p_grid = np.linspace(0, 1, 501)

    # --- 3. Calculate the policy function for the Myopic Regulator ---
    print("\n[1/2] Calculating Myopic Policy Function...")
    # Combine all parameters required for the myopic model
    myopic_params_full = {**SHARED_PARAMS, **MYOPIC_EXTRA_PARAMS}

    # The a_values_to_test should be consistent with p_grid for the best match
    a_grid_myopic = np.linspace(0, 1, 501)

    policy_myopic = calculate_policy_function(
        p_values=p_grid,
        ai_values=a_grid_myopic,
        static_params=myopic_params_full
    )
    print("Myopic Policy calculation complete.")

    # --- 4. Calculate the policy function for the Dynamic Regulator ---
    print("\n[2/2] Calculating Dynamic Policy Function...")
    optimizer = GeneticOptimizer(
        h=SHARED_PARAMS['h'],
        Delta=SHARED_PARAMS['Delta'],
        beta=DP_EXTRA_PARAMS['beta'],
        c=DP_EXTRA_PARAMS['c'],
        cost_type=DP_EXTRA_PARAMS['cost_type']
    )

    # Run value function iteration to solve the model.
    # Set verbose=False to keep the output clean.
    optimizer.value_function_iteration(verbose=False, tol=1e-6)
    policy_dynamic = optimizer.policy
    # The p-grid for the DP model is retrieved from the optimizer object to ensure consistency
    p_grid_dynamic = optimizer.p_grid
    print("Dynamic Policy calculation complete.")

    # --- 5. Plot the comparison ---
    print("\nPlotting results...")
    plt.figure(figsize=(8, 8))

    # Plot the Myopic Policy
    plt.plot(p_grid, policy_myopic, 'b-', linewidth=2.5, label='Myopic Regulator')

    # Plot the Dynamic Policy
    plt.plot(p_grid_dynamic, policy_dynamic, 'r--', linewidth=2.5, label='Forward-looking Regulator')

    # Plot the y=x reference line
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='a = p (No Intervention)')

    # Set the chart title and labels
    title_str = (f'Comparison of Policy Functions\n'
                 f'Shared parameters: h={SHARED_PARAMS["h"]}, Δ={SHARED_PARAMS["Delta"]}\n'
                 f'Forward-looking parameter: β={DP_EXTRA_PARAMS["beta"]}')
    plt.title(title_str, fontsize=14)
    plt.xlabel("p (State Variable: Allele A Frequency)", fontsize=12)
    plt.ylabel("a (Control Variable: Optimal Policy)", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.gca().set_aspect('equal', adjustable='box')  # Makes x and y axes equal length
    plt.tight_layout()

    # Save the figure
    os.makedirs("comparison_results", exist_ok=True)
    output_path = os.path.join("comparison_results", "policy_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    run_comparison()