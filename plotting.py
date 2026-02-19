# plotting.py
# Unified plotting module containing all visualization functions

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d


# =============================================================================
#  Robust Simulation Plots
# =============================================================================

def create_plots(results, output_dir):
    """
    Generate all plots from Robust Simulation comparative statics results.
    """
    print(f"\nGenerating plots, saving to '{output_dir}'...")

    sweep_param_name = results["sweep_param_name"]
    sweep_param_values = results["sweep_param_values"]
    ai_values = results["ai_values"]
    potential_benefits = results["potential_benefits"]
    optimal_ai_proportions = results["optimal_ai_proportions"]
    avg_yields_mesh = results["avg_yields_mesh"]

    xlabel_map = {
        'h': 'h',
        'Delta': 'Δ',
        'pi': 'π',
        'gamma': 'γ',
        'num_generations': 'Policy Duration (Generations)',
        'initial_A_proportion': r'$p_0$'
    }
    Y_label = xlabel_map.get(sweep_param_name, sweep_param_name)

    # Plot 1: Potential Benefit (percentage scale, fixed y-axis 0%–15%)
    import matplotlib.ticker as mticker
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_param_values, potential_benefits * 100, 'b-', linewidth=2)
    plt.xlabel(Y_label, fontsize=18)
    plt.ylabel('Potential Benefit (%)', fontsize=18)
    plt.title(f'Potential Benefit vs. {Y_label}', fontsize=22)
    plt.ylim(0, 15)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"benefit_vs_{sweep_param_name}.png"), dpi=300)

    # Plot 2: Optimal AI Proportion
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_param_values, optimal_ai_proportions, 'b-', linewidth=2)
    plt.xlabel(Y_label, fontsize=18)
    plt.ylabel('Optimal a', fontsize=18)
    plt.title(f'Optimal a vs. {Y_label}', fontsize=22)
    plt.ylim(0,1)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"optimal_a_vs_{sweep_param_name}.png"), dpi=300)

    # Plot 3: 3D Surface
    X_MESH, Y_MESH = np.meshgrid(ai_values, sweep_param_values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_MESH, Y_MESH, avg_yields_mesh, cmap='viridis', edgecolor='none')
    ax.set_xlabel('a', fontsize=16, labelpad=10)
    ax.set_ylabel(Y_label, fontsize=16, labelpad=10)
    ax.set_zlabel('Average Yield', fontsize=16, labelpad=10)
    ax.set_title(f'Average Yield vs. a and {Y_label}', fontsize=22, pad=20)
    ax.tick_params(axis='both', labelsize=13)
    # cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    # cbar.set_label('Average Yield', fontsize=18)
    # cbar.ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"yield_surface_vs_{sweep_param_name}.png"), dpi=300)

    # Plot 4: Contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_MESH, Y_MESH, avg_yields_mesh, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Average Yield', fontsize=18)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel('a', fontsize=18)
    plt.ylabel(Y_label, fontsize=18)
    plt.title(f'Contour Plot: Yield vs. a and {Y_label}', fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"yield_contour_vs_{sweep_param_name}.png"), dpi=300)

    print("All plots generated and saved as PNG files.")
    plt.show()


# =============================================================================
#  Myopic Simulation Plots
# =============================================================================

def plot_p_evolution_comparison(results_list, initial_p_list, static_params, output_dir):
    """
    Plot comparison of dynamic evolution paths under different initial p values.
    """
    print(f"\nGenerating multi-initial-value evolution comparison, saving to '{output_dir}'...")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results_list)))

    if not results_list:
        print("Warning: No simulation results provided for plotting.")
        return

    total_generations = 0
    for i, results in enumerate(results_list):
        total_generations = results['total_generations']
        initial_p = initial_p_list[i]

        allele_p_series = results['allele_p_series']
        optimal_a_series = results['optimal_a_series']

        generations_axis = np.arange(1, total_generations + 1)

        plt.plot(generations_axis, allele_p_series, color=colors[i], marker='o',
                 markersize=4, linestyle='-', linewidth=2,
                 label=f'Initial p = {initial_p:.1f}')

        plt.plot(generations_axis, optimal_a_series, color=colors[i], marker='x',
                 markersize=5, linestyle='--', linewidth=2,
                 label=f'a (Initial p={initial_p:.1f})')

    h = static_params.get('h', 'N/A')
    Delta = static_params.get('Delta', 'N/A')

    plt.title(f'Dynamics of p and a under Myopic Regulation\n'
              f'(h={h}, Δ={Delta})', fontsize=22)
    plt.xlabel('Generation', fontsize=18)
    plt.ylabel('p or a', fontsize=18)
    plt.ylim(0, 1)
    plt.xlim(left=0.5, right=total_generations + 0.5)
    plt.legend(title="Initial Conditions", fontsize=14, title_fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    if total_generations < 25:
        plt.xticks(np.arange(1, total_generations + 1, step=1))
    else:
        plt.xticks(np.arange(0, total_generations + 1,
                              step=max(1, total_generations // 10)))

    plt.tight_layout()
    filepath = os.path.join(output_dir, "dynamic_p_evolution_comparison.png")
    plt.savefig(filepath, dpi=300)
    print(f"Evolution comparison plot saved to: {filepath}")
    plt.show()


def plot_policy_function(p_values, optimal_policy, model_params, output_dir):
    """
    Plot the optimal policy function a*(p).
    """
    plt.figure(figsize=(7, 7))
    plt.plot(p_values, optimal_policy, 'b-', linewidth=2, label='Optimal Policy a*(p)')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='a = p (no intervention)')
    plt.title(
        f'Myopic Optimal Policy Function a*(p)\n'
        f'(h={model_params["h"]}, Δ={model_params["Delta"]})',
        fontsize=20
    )
    plt.xlabel("p (Frequency of Allele A)", fontsize=18)
    plt.ylabel("a* (Optimal Policy)", fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "policy_function_a_vs_p.png"), dpi=300)
    plt.show()


def plot_yield_curves(p_values_to_show, ai_values, static_params, results_dir):
    """
    Plot "expected yield vs. policy a" curves for selected p values,
    marking the optimal point on each curve.
    """
    from core import compute_expected_yield

    print(f"\n--- Generating yield curve explanation for p = {p_values_to_show} ---")
    plt.figure(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(p_values_to_show)))

    Delta = static_params['Delta']
    h = static_params['h']
    C1 = static_params.get('C1', 1.0)
    C2 = static_params.get('C2', 1.0)
    n_exp = static_params.get('n', 2)
    m_exp = static_params.get('m', 2)

    for i, p_val in enumerate(p_values_to_show):
        expected_yields = []
        for a_candidate in ai_values:
            yield_val = compute_expected_yield(
                p_val, a_candidate, Delta, h, C1, C2, n_exp, m_exp
            )
            expected_yields.append(yield_val)

        max_yield_idx = np.argmax(expected_yields)
        optimal_a = ai_values[max_yield_idx]
        max_yield = expected_yields[max_yield_idx]

        plt.plot(ai_values, expected_yields, color=colors[i],
                 label=f'When p = {p_val:.1f}')
        plt.plot(optimal_a, max_yield, 's', color='red', markersize=6,
                 markerfacecolor='none', markeredgecolor='red',
                 label=f'Optimal Choice a*({p_val:.2f}) ≈ {optimal_a:.2f}')

    plt.title('Maximizing Expected Yield of the Current Generation', fontsize=22)
    plt.xlabel('Control Variable: a (Allele G in Controlled Breeding)', fontsize=18)
    plt.ylabel('Expected Yield of the Current Generation', fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()

    output_filepath = os.path.join(results_dir, "yield_curve_explanation.png")
    plt.savefig(output_filepath, dpi=300)
    print(f"Yield curve explanation plot saved to: {output_filepath}")
    plt.show()


# =============================================================================
#  DP Simulation Plots
# =============================================================================

def plot_dp_results(optimizer, p0_list=[0.2, 0.5, 0.8], output_dir=None):
    """
    Plot DP model four-panel results figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    # 1. Value Function
    axes[0, 0].plot(optimizer.p_grid, optimizer.V, 'b-', linewidth=2)
    axes[0, 0].set_ylim( 0, 25)
    axes[0, 0].set_xlabel('p ', fontsize=16)
    axes[0, 0].set_ylabel('V(p)', fontsize=16)
    axes[0, 0].set_title('Value Function', fontsize=18)
    axes[0, 0].tick_params(axis='both', labelsize=13)
    axes[0, 0].grid(True)

    # 2. Optimal Policy
    axes[0, 1].plot(optimizer.p_grid, optimizer.policy, 'r-', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel('p', fontsize=16)
    axes[0, 1].set_ylabel('Optimal a', fontsize=16)
    axes[0, 1].set_title('Optimal Policy Function', fontsize=18)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].tick_params(axis='both', labelsize=13)
    axes[0, 1].grid(True)

    # 3. Trajectory of p
    colors = ['blue', 'green', 'red']
    for i, p0 in enumerate(p0_list):
        p_traj, a_traj = optimizer.simulate_trajectory(p0, T=10)
        axes[1, 0].plot(p_traj, color=colors[i], linewidth=2,
                        label=f'Initial p={p0}', marker='o', markersize=3)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel('Generation', fontsize=16)
    axes[1, 0].set_ylabel(r'$p_t$', fontsize=16)
    axes[1, 0].set_title('Dynamic trajectory of p', fontsize=18)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].tick_params(axis='both', labelsize=13)
    axes[1, 0].grid(True)

    # 4. Trajectory of a
    for i, p0 in enumerate(p0_list):
        p_traj, a_traj = optimizer.simulate_trajectory(p0, T=10)
        axes[1, 1].plot(a_traj, color=colors[i], linewidth=2,
                        label=f'Initial p={p0}', marker='s', markersize=3)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel('Generation', fontsize=16)
    axes[1, 1].set_ylabel(r'$a_t$', fontsize=16)
    axes[1, 1].set_title('Dynamics trajectory of a', fontsize=18)
    axes[1, 1].legend(fontsize=14)
    axes[1, 1].tick_params(axis='both', labelsize=13)
    axes[1, 1].grid(True)

    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "dp_results.png"), dpi=300)

    plt.show()



def plot_bellman_objective(optimizer, p_values_to_plot=[0.2, 0.5, 0.8],
                           output_dir=None):
    """
    Visualize the Bellman equation RHS objective function.
    """
    a_grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(10, 7))

    V_interpolator = interp1d(optimizer.p_grid, optimizer.V, kind='cubic',
                              fill_value="extrapolate")

    for p_val in p_values_to_plot:
        objective_values = []
        for a in a_grid:
            payoff = optimizer.immediate_payoff(a, p_val)
            next_p = optimizer.state_transition(a, p_val)
            future_value = V_interpolator(next_p)
            total_value = payoff + optimizer.beta * future_value
            objective_values.append(total_value)

        plt.plot(a_grid, objective_values, label=f'p = {p_val:.2f}')

        max_val_idx = np.argmax(objective_values)
        optimal_a = a_grid[max_val_idx]
        max_val = objective_values[max_val_idx]
        plt.plot(optimal_a, max_val, 'o', markersize=8,
                 label=f'a*({p_val:.2f}) ≈ {optimal_a:.2f}')

    plt.xlabel('a (Control Variable)', fontsize=18)
    plt.ylabel(r'$R(a, p) + \beta V(p\')$', fontsize=18)
    plt.title('Bellman Equation Objective Function', fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.legend(fontsize=14)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "bellman_objective.png"), dpi=300)

    plt.show()


# =============================================================================
#  Policy Comparison Plot
# =============================================================================

def plot_policy_comparison(p_grid_myopic, policy_myopic,
                           p_grid_dp, policy_dp,
                           shared_params, dp_params, output_dir):
    """
    Plot Myopic vs. Forward-looking policy function comparison.
    """
    plt.figure(figsize=(8, 8))

    plt.plot(p_grid_myopic, policy_myopic, 'b-', linewidth=2.5,
             label='Myopic Regulator')
    plt.plot(p_grid_dp, policy_dp, 'r--', linewidth=2.5,
             label='Forward-looking Regulator')
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='a = p')

    title_str = (
        f'Comparison of Policy Functions\n'
        f'Shared: h={shared_params["h"]}, Δ={shared_params["Delta"]}\n'
        f'Dynamic: β={dp_params["beta"]}'
    )
    plt.title(title_str, fontsize=20)
    plt.xlabel("p (State Variable)", fontsize=18)
    plt.ylabel("a (Control Variable)", fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "policy_comparison.png"), dpi=300)
    print(f"Comparison plot saved to: {output_dir}/policy_comparison.png")
    plt.show()
