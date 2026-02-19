# run_robust.py
# Interface 1: Robust Simulation (Monte Carlo)
# Comparative statics analysis via command-line parameter selection

import numpy as np
import argparse
import sys
import datetime
import os

from core import perform_comparative_statics
from plotting import create_plots

# --- Analysis configurations ---
ANALYSIS_CONFIGS = {
    'h': {'name': 'h', 'values': np.linspace(0, 1, 101)},
    'Delta': {'name': 'Delta', 'values': np.linspace(0.1, 0.5, 41)},
    'pi': {'name': 'pi', 'values': np.linspace(0, 1, 101)},
    'gamma': {'name': 'gamma', 'values': np.linspace(0, 1, 101)},
    'num_generations': {'name': 'num_generations', 'values': np.arange(2, 50, 1, dtype=int)},
    'initial_A_proportion': {'name': 'initial_A_proportion', 'values': np.linspace(0, 1, 101)},
}

RESULTS_DIR = "results_robust"


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Run Robust (Monte Carlo) Simulation or plot saved results."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # simulate command
    parser_sim = subparsers.add_parser('simulate', help='Run a new simulation and save results.')
    parser_sim.add_argument(
        'param', choices=ANALYSIS_CONFIGS.keys(),
        help=f"Parameter to sweep. Available: {list(ANALYSIS_CONFIGS.keys())}"
    )

    # plot command
    parser_plot = subparsers.add_parser('plot', help='Load saved results and generate plots.')
    parser_plot.add_argument('filepath', help="Path to .npy results file.")

    args = parser.parse_args()

    if args.command == 'simulate':
        config = ANALYSIS_CONFIGS[args.param]
        analysis_param_name = config['name']
        analysis_param_values = config['values']

        print(f"--- Robust Simulation ---")
        print(f"Target parameter: {analysis_param_name}")

        # Base parameters (disease-effect model: pi and gamma)
        BASE_PARAMS = {
            "population_size": 10000,
            "num_generations": 10,
            "initial_A_proportion": 0.5,
            "h": 0.8,
            "Delta": 0.3,
            "pi": 0.3,
            "gamma": 0.8,
        }
        ai_values_to_scan = np.linspace(0.5, 1, 51)

        static_params_for_run = BASE_PARAMS.copy()
        if analysis_param_name in static_params_for_run:
            del static_params_for_run[analysis_param_name]

        analysis_results = perform_comparative_statics(
            static_params=static_params_for_run,
            sweep_param_name=analysis_param_name,
            sweep_param_values=analysis_param_values,
            ai_values=ai_values_to_scan
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"results_{analysis_param_name}_{timestamp}.npy"
        output_filepath = os.path.join(RESULTS_DIR, filename)
        np.save(output_filepath, analysis_results)
        print(f"\nSimulation complete. Results saved to: {output_filepath}")

    elif args.command == 'plot':
        print(f"--- Plot mode ---")
        print(f"Loading results file: {args.filepath}")
        try:
            loaded_results = np.load(args.filepath, allow_pickle=True).item()
            create_plots(loaded_results, RESULTS_DIR)
        except FileNotFoundError:
            print(f"Error: File '{args.filepath}' not found.")
            sys.exit(1)
