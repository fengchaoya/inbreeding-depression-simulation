# main.py (Final version, supports standalone simulation runs and plotting)

import numpy as np
import argparse  # Import the library for command-line argument parsing
import sys
import datetime
import os

# Import our custom modules
from module_simulation import perform_comparative_statics
from module_plot import create_plots

# --- 1. Centralize all analysis configurations ---
# We store the configuration for each analysis (parameter name and value range)
# in a dictionary for easy access.
ANALYSIS_CONFIGS = {
    'h': {
        'name': 'h',
        'values': np.linspace(0, 1, 101)
    },
    'Delta': {
        'name': 'Delta',
        'values': np.linspace(0.1, 0.5, 41)
    },
    'pi': {
        'name': 'pi',
        'values': np.linspace(0, 1, 101)
    },
    'gamma': {
        'name': 'gamma',
        'values': np.linspace(0, 1, 101)
    },
    'num_generations': {
        'name': 'num_generations',
        'values': np.arange(2, 50, 1, dtype=int)
    },
    'initial_A_proportion': {
        'name': 'initial_A_proportion',
        'values': np.linspace(0, 1, 101)
    }
}

# --- 2. Define global constants ---
RESULTS_DIR = "results"  # Define the name of the results folder

if __name__ == '__main__':
    # --- 3. Ensure the results folder exists at the start of the program ---
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Create the folder if it doesn't exist; do nothing if it does.

    # --- 4. Set up the command-line argument parser ---
    parser = argparse.ArgumentParser(
        description="Run inbreeding depression simulations or plot results from a file.",
        epilog="Use 'python main.py simulate -h' or 'python main.py plot -h' for more specific help."
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Choose a command to execute')

    # Create the parser for the 'simulate' command
    parser_simulate = subparsers.add_parser('simulate', help='Run a new simulation and save the results.')
    parser_simulate.add_argument(
        'param',
        choices=ANALYSIS_CONFIGS.keys(),
        help=f"Select the parameter for comparative statics analysis. Available options: {list(ANALYSIS_CONFIGS.keys())}"
    )

    # Create the parser for the 'plot' command
    parser_plot = subparsers.add_parser('plot', help='Load simulation results from a file and generate plots.')
    parser_plot.add_argument(
        'filepath',
        help="Path to the .npy file containing simulation results (e.g., results_h.npy)."
    )

    args = parser.parse_args()

    # --- 5. Execute the corresponding action based on the command ---

    if args.command == 'simulate':
        # --- Execute Simulation ---
        config = ANALYSIS_CONFIGS[args.param]
        analysis_param_name = config['name']
        analysis_param_values = config['values']

        print(f"--- Mode: Simulate ---")
        print(f"Target parameter: {analysis_param_name}")

        BASE_PARAMS = {
            "population_size": 10000, "num_generations": 25, "initial_A_proportion": 0.5,
            "h": 0.8, "Delta": 0.3, "pi": 0.3, "gamma": 0.8
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

        # Get the current time and format it as a string
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Use os.path.join to build a cross-platform compatible file path
        filename = f"results_{analysis_param_name}_{timestamp}.npy"
        output_filepath = os.path.join(RESULTS_DIR, filename)

        np.save(output_filepath, analysis_results)
        print(f"\nSimulation complete! Results have been saved to: {output_filepath}")

    elif args.command == 'plot':
        # --- Execute Plotting ---
        print(f"--- Mode: Plot ---")
        print(f"Loading results file: {args.filepath}")

        try:
            loaded_results = np.load(args.filepath, allow_pickle=True).item()
            # Also pass the path of the results directory to the plotting function
            create_plots(loaded_results, RESULTS_DIR)
        except FileNotFoundError:
            print(f"Error: File '{args.filepath}' not found. Please check the path and filename.")
            sys.exit(1)

# The commented-out section below contains the old, manual way of running the script.
# It has been replaced by the command-line interface above.
# I am leaving it here commented out for your reference, but it can be safely deleted.
#
# # =============================================================================
# #  MAIN EXECUTION BLOCK (OLD VERSION)
# # =============================================================================
#
# if __name__ == '__main__':
#     # --- 1. Define "base" or "default" values for all parameters ---
#     BASE_PARAMS = {
#         "population_size": 10000,
#         "num_generations": 10,
#         "initial_A_proportion": 0.5,
#         "h": 0.8,              # Dominance coefficient
#         "Delta": 0.3,          # Selection advantage
#         "pi": 0.3,             # Disease incidence
#         "gamma": 0.8           # Disease severity
#     }
#
#     # Define the optimization range for 'a' (this usually remains constant across analyses)
#     ai_values_to_scan = np.linspace(0.5, 1, 51)
#
#     # --- 2. Select and configure the analysis you want to run ---
#     # --- Just uncomment the block for the analysis you wish to perform ---
#
#     # [Analysis Case 1: Comparative statics for h]
#     # analysis_param_name = 'h'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # [Analysis Case 2: Comparative statics for Delta (selection advantage)]
#     # analysis_param_name = 'Delta'
#     # analysis_param_values = np.linspace(0.1, 0.5, 41)
#
#     # [Analysis Case 3: Comparative statics for pi (disease incidence)]
#     # analysis_param_name = 'pi'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # [Analysis Case 4: Comparative statics for gamma (disease severity)] (New)
#     # analysis_param_name = 'gamma'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # [Analysis Case 5: Comparative statics for num_generations] (New)
#     # analysis_param_name = 'num_generations'
#     # # Using np.arange is more intuitive for integer parameters
#     # analysis_param_values = np.arange(2, 50, 1, dtype=int)
#
#     # [Analysis Case 6: Comparative statics for initial_A_proportion] (New)
#     # analysis_param_name = 'initial_A_proportion'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # --- 3. Prepare parameters and execute ---
#
#     # Create a copy of the base parameters to serve as the static parameters for this run
#     static_params_for_run = BASE_PARAMS.copy()
#     # Remove the entry for the parameter that is about to be swept from the "static" dictionary
#     if analysis_param_name in static_params_for_run:
#         del static_params_for_run[analysis_param_name]
#
#     # Execute the generic analysis function
#     analysis_results = perform_comparative_statics(
#         static_params=static_params_for_run,
#         sweep_param_name=analysis_param_name,
#         sweep_param_values=analysis_param_values,
#         ai_values=ai_values_to_scan
#     )
#
#     # --- 4. Plotting ---
#     create_plots(analysis_results)