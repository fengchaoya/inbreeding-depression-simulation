# Genetic Breeding Strategy Optimization

A simulation framework for analyzing optimal artificial insemination (AI) strategies in livestock breeding under inbreeding depression. The project compares three regulatory perspectives: robust (Monte Carlo) simulation, myopic (single-period) optimization, and forward-looking (dynamic programming) optimization.

## Project Structure

```
├── core.py               # Core library: loss function, yield calculation, simulation engines, DP solver
├── plotting.py           # All visualization functions
├── run_robust.py         # Interface 1: Robust (Monte Carlo) simulation
├── run_myopic.py         # Interface 2: Myopic regulator simulation
├── run_dp.py             # Interface 3: Forward-looking (DP) regulator simulation
├── compare_policies.py   # Compare myopic vs. forward-looking policy functions
└── requirements.txt      # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Concepts

The model tracks a population of diploid organisms with two alleles at a single locus: **A** (high-yield) and **a** (wild-type). A regulator chooses `a_t` — the proportion of allele A used in controlled breeding each generation — to maximize total yield while managing inbreeding costs.

**Key variables:**

| Symbol | Description |
|--------|-------------|
| `p`    | Frequency of allele A in the maternal pool |
| `a`    | Proportion of allele A in controlled breeding (policy variable) |
| `Δ`    | Yield advantage of allele A |
| `h`    | Dominance coefficient of heterozygotes |

**Unified loss function** (defined once in `core.py`, used by all simulations):

```
Loss = C1 × (AA_prop)^n + C2 × (aa_prop)^m
```

Parameters `C1`, `C2`, `n`, `m` can be adjusted in a single location to affect all simulations consistently.

## Usage

### Interface 1: Robust Simulation (`run_robust.py`)

Runs full Monte Carlo simulations with individual-level genotype tracking. Useful for comparative statics analysis — sweeping one parameter while holding others fixed.

```bash
# Run a simulation sweeping parameter h
python run_robust.py simulate h

# Run a simulation sweeping parameter Delta
python run_robust.py simulate Delta

# Plot results from a saved file
python run_robust.py plot results_robust/results_h_20260218-143000.npy
```

Available sweep parameters: `h`, `Delta`, `num_generations`, `initial_A_proportion`.

Model parameters (including loss function coefficients) are configured in the `BASE_PARAMS` dictionary inside the script.

### Interface 2: Myopic Simulation (`run_myopic.py`)

Analyzes the myopic (single-period) regulator who optimizes yield one generation at a time using analytical expected values. Includes three tasks controlled by boolean flags at the top of the script:

```bash
python run_myopic.py
```

| Flag | Task |
|------|------|
| `RUN_POLICY_FUNCTION_ANALYSIS` | Compute and plot the optimal policy function a*(p) |
| `RUN_DYNAMIC_SIMULATION` | Simulate p and a dynamics from multiple initial conditions |
| `RUN_YIELD_CURVE_EXPLANATION` | Plot yield vs. a curves for selected p values |

Parameters are configured in the `MODEL_PARAMS` dictionary inside the script.

### Interface 3: Forward-Looking Simulation (`run_dp.py`)

Solves the infinite-horizon dynamic programming problem via value function iteration. The forward-looking regulator accounts for how today's breeding decision affects future genetic composition.

```bash
python run_dp.py
```

Outputs include the value function, optimal policy function, and simulated trajectories from multiple initial conditions. Parameters are set when instantiating `GeneticOptimizer` at the top of the script.

### Policy Comparison (`compare_policies.py`)

Computes both policy functions under identical genetic parameters and plots them side by side.

```bash
python compare_policies.py
```

## Configuring the Loss Function

All simulations share the same loss function form. To change it, modify the `C1`, `C2`, `n`, `m` parameters in the relevant script's parameter dictionary:

```python
# In run_robust.py, run_myopic.py, or compare_policies.py:
"C1": 2.0,   # Multiplier on AA homozygote term
"C2": 2.0,   # Multiplier on aa homozygote term
"n": 2,      # Exponent on AA homozygote term
"m": 2,      # Exponent on aa homozygote term

# In run_dp.py (passed to GeneticOptimizer constructor):
GeneticOptimizer(h=0.6, Delta=0.4, beta=0.95, C1=2.0, C2=2.0, n=2, m=2)
```

## Output

Each simulation script saves results (`.npy` files and `.png` plots) to its own output directory:

| Script | Output Directory |
|--------|-----------------|
| `run_robust.py` | `results_robust/` |
| `run_myopic.py` | `results_myopic/` |
| `run_dp.py` | `results_dp/` |
| `compare_policies.py` | `results_comparison/` |
