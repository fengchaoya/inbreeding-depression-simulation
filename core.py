# core.py
# Unified core library
# Contains: loss function, yield calculation, simulation engines, policy function, DP solver

import numpy as np
import time
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
#  Unified Loss Function (single definition for the entire project)
# =============================================================================

def loss_function(a, p, C1=1.0, C2=1.0, n=2, m=2):
    """
    Unified loss function.

    Loss = C1 * (a * p)^n + C2 * ((1-a) * (1-p))^m

    Args:
        a (float): Proportion of allele 'A' in controlled breeding.
        p (float): Current frequency of allele 'A'.
        C1 (float): Multiplier for the first term (default 1.0).
        C2 (float): Multiplier for the second term (default 1.0).
        n (int/float): Exponent for the first term (default 2).
        m (int/float): Exponent for the second term (default 2).

    Returns:
        float: Loss value.
    """
    return C1 * (a * p) ** n + C2 * ((1 - a) * (1 - p)) ** m


# =============================================================================
#  Yield Calculation — Robust Simulation (disease-effect model)
# =============================================================================

def calculate_yield_robust(aa_prop, Aa_prop, AA_prop, Delta, h, pi, gamma):
    """
    Calculate per-generation yield for the Robust (Monte Carlo) simulation.

    Uses the disease-effect model:
        disease_effect = 1 - pi * gamma
    Homozygotes (aa and AA) are affected by disease; heterozygotes (Aa) are not.

    Args:
        aa_prop, Aa_prop, AA_prop (float): Genotype frequencies.
        Delta, h, pi, gamma (float): Model parameters.

    Returns:
        float: Total yield for the current generation.
    """
    disease_effect = 1 - pi * gamma

    aa_yield = aa_prop * 1.0 * disease_effect
    Aa_yield = Aa_prop * (1.0 + h * Delta)  # Heterozygotes unaffected by disease
    AA_yield = AA_prop * (1.0 + Delta) * disease_effect

    return aa_yield + Aa_yield + AA_yield


# =============================================================================
#  Yield Calculation — Myopic & DP Simulation (parameterized loss model)
# =============================================================================

def calculate_yield_analytical(aa_prop, Aa_prop, AA_prop, Delta, h,
                               C1=2.0, C2=2.0, n=2, m=2):
    """
    Calculate per-generation yield for Myopic and DP simulations.

    Uses a parameterized loss function:
        Loss = C1 * AA_prop^n + C2 * aa_prop^m

    Args:
        aa_prop, Aa_prop, AA_prop (float): Genotype frequencies.
        Delta, h (float): Model parameters.
        C1, C2 (float): Loss function multipliers (default 2.0).
        n, m (int/float): Loss function exponents (default 2).

    Returns:
        float: Total yield for the current generation.
    """
    loss = C1 * (AA_prop ** n) + C2 * (aa_prop ** m)

    aa_yield = aa_prop * 1.0
    Aa_yield = Aa_prop * (1.0 + h * Delta)
    AA_yield = AA_prop * (1.0 + Delta)

    return aa_yield + Aa_yield + AA_yield - loss


def compute_expected_yield(p, a, Delta, h, C1=2.0, C2=2.0, n=2, m=2):
    """
    Analytically compute expected yield given allele frequency p and policy a.

    Convenience function shared by myopic and DP models, eliminating
    repeated "compute genotype frequencies -> compute yield" logic.

    Args:
        p (float): Current allele 'A' frequency (maternal pool).
        a (float): Proportion of allele 'A' in controlled breeding.
        Delta, h (float): Model parameters.
        C1, C2, n, m: Loss function parameters.

    Returns:
        float: Expected yield.
    """
    p_a_maternal = 1 - p  # Frequency of allele 'a'
    p_A_maternal = p

    p_a_paternal = 1 - a
    p_A_paternal = a

    E_prop_aa = p_a_maternal * p_a_paternal
    E_prop_AA = p_A_maternal * p_A_paternal
    E_prop_Aa = 1 - E_prop_aa - E_prop_AA

    return calculate_yield_analytical(
        E_prop_aa, E_prop_Aa, E_prop_AA, Delta, h, C1, C2, n, m
    )


# =============================================================================
#  Monte Carlo Simulation: Genetic Logic (for Robust Simulation, logic unchanged)
# =============================================================================

def initialize_population(pop_size, initial_A_proportion):
    """
    Initialize population. 1 represents 'A', 0 represents 'a'.
    """
    num_alleles = 2 * pop_size
    num_A_alleles = int(num_alleles * initial_A_proportion)

    alleles = np.zeros(num_alleles, dtype=np.int8)
    alleles[:num_A_alleles] = 1

    np.random.shuffle(alleles)
    return alleles.reshape((pop_size, 2))


def calculate_genotype_proportions(population):
    """
    Efficiently compute genotype frequencies.
    """
    pop_size = population.shape[0]
    allele_sums = np.sum(population, axis=1)

    aa_prop = np.sum(allele_sums == 0) / pop_size
    Aa_prop = np.sum(allele_sums == 1) / pop_size
    AA_prop = np.sum(allele_sums == 2) / pop_size

    return aa_prop, Aa_prop, AA_prop


def create_offspring(parents, pop_size, ai_proportion):
    """
    Create offspring using "random dam + controlled sire" rule (vectorized).
    """
    mother_indices = np.random.randint(0, pop_size, size=pop_size)
    mothers = parents[mother_indices, :]

    maternal_gamete_indices = np.random.randint(0, 2, size=pop_size)
    maternal_gametes = mothers[np.arange(pop_size), maternal_gamete_indices]

    paternal_gametes = np.random.choice(
        [1, 0], size=pop_size, p=[ai_proportion, 1 - ai_proportion]
    )

    return np.column_stack((maternal_gametes, paternal_gametes))


# =============================================================================
#  Monte Carlo Simulation Engine (Robust Simulation core, logic unchanged)
# =============================================================================

def run_single_simulation(population_size, num_generations, initial_A_proportion,
                          ai_proportion, Delta, h, pi, gamma):
    """
    Core engine: run one complete Monte Carlo simulation and return average yield.
    Uses the disease-effect model (calculate_yield_robust).
    """
    population = initialize_population(population_size, initial_A_proportion)
    total_yield = 0
    for _ in range(num_generations):
        aa_prop, Aa_prop, AA_prop = calculate_genotype_proportions(population)
        gen_yield = calculate_yield_robust(
            aa_prop, Aa_prop, AA_prop, Delta, h, pi, gamma
        )
        total_yield += gen_yield
        population = create_offspring(population, population_size, ai_proportion)

    return total_yield / num_generations


def perform_comparative_statics(static_params, sweep_param_name,
                                sweep_param_values, ai_values):
    """
    General comparative statics analysis function (Monte Carlo version).
    """
    print(f"Starting comparative statics for parameter '{sweep_param_name}'...")
    start_time = time.time()

    num_sweep = len(sweep_param_values)
    avg_yields_mesh = np.zeros((num_sweep, len(ai_values)))
    potential_benefits = np.zeros(num_sweep)
    optimal_ai_proportions = np.zeros(num_sweep)

    dynamic_params = static_params.copy()

    for i, sweep_val in enumerate(sweep_param_values):
        dynamic_params[sweep_param_name] = sweep_val
        avg_yields_for_current_sweep = np.zeros(len(ai_values))

        for j, ai_prop in enumerate(ai_values):
            dynamic_params['ai_proportion'] = ai_prop
            avg_yields_for_current_sweep[j] = run_single_simulation(**dynamic_params)

        avg_yields_mesh[i, :] = avg_yields_for_current_sweep

        max_yield = np.max(avg_yields_for_current_sweep)
        max_index = np.argmax(avg_yields_for_current_sweep)
        yield_at_ai_1 = avg_yields_for_current_sweep[-1]

        potential_benefits[i] = max_yield - yield_at_ai_1
        optimal_ai_proportions[i] = ai_values[max_index]

        print(f"  {sweep_param_name} = {sweep_val:.3f} done "
              f"(progress: {i + 1}/{num_sweep})")

    end_time = time.time()
    print(f"Analysis complete. Total time: {end_time - start_time:.2f} sec.")

    return {
        "sweep_param_name": sweep_param_name,
        "sweep_param_values": sweep_param_values,
        "ai_values": ai_values,
        "avg_yields_mesh": avg_yields_mesh,
        "potential_benefits": potential_benefits,
        "optimal_ai_proportions": optimal_ai_proportions,
        "static_params": static_params
    }


# =============================================================================
#  Myopic Dynamic Simulation (analytical version, logic unchanged)
# =============================================================================

def run_dynamic_policy_simulation(total_generations, ai_values, static_params):
    """
    Run dynamic policy simulation, finding the optimal 'a' each generation
    (myopic strategy). Uses analytical computation, not Monte Carlo.
    """
    print("--- Starting dynamic policy simulation ---")
    start_time = time.time()

    pop_size = static_params['population_size']
    initial_A_prop = static_params['initial_A_proportion']
    h = static_params['h']
    Delta = static_params['Delta']
    C1 = static_params.get('C1', 1.0)
    C2 = static_params.get('C2', 1.0)
    n_exp = static_params.get('n', 2)
    m_exp = static_params.get('m', 2)

    current_population = initialize_population(pop_size, initial_A_prop)

    optimal_a_series = []
    genotype_proportions_series = []
    allele_p_series = []

    for gen in range(total_generations):
        prop_aa, prop_Aa, prop_AA = calculate_genotype_proportions(current_population)
        p_a_maternal = prop_aa + 0.5 * prop_Aa
        p_A_maternal = 1 - p_a_maternal

        expected_yields = []
        for a_candidate in ai_values:
            yield_val = compute_expected_yield(
                p_A_maternal, a_candidate, Delta, h, C1, C2, n_exp, m_exp
            )
            expected_yields.append(yield_val)

        optimal_a_t = ai_values[np.argmax(expected_yields)]

        optimal_a_series.append(optimal_a_t)
        genotype_proportions_series.append((prop_aa, prop_Aa, prop_AA))
        allele_p_series.append(p_A_maternal)

        current_population = create_offspring(current_population, pop_size, optimal_a_t)

        if (gen + 1) % 10 == 0:
            print(f"  Generation {gen + 1}/{total_generations} done.")

    end_time = time.time()
    print(f"Dynamic simulation complete. Total time: {end_time - start_time:.2f} sec.")

    return {
        "total_generations": total_generations,
        "optimal_a_series": np.array(optimal_a_series),
        "genotype_proportions_series": np.array(genotype_proportions_series),
        "allele_p_series": np.array(allele_p_series)
    }


# =============================================================================
#  Myopic Policy Function Calculation (analytical, single copy)
# =============================================================================

def calculate_policy_function(p_values, ai_values, static_params):
    """
    Compute the optimal myopic policy a* for each allele frequency p.
    """
    print("--- Computing optimal policy function a*(p) ---")

    Delta = static_params['Delta']
    h = static_params['h']
    C1 = static_params.get('C1', 1.0)
    C2 = static_params.get('C2', 1.0)
    n_exp = static_params.get('n', 2)
    m_exp = static_params.get('m', 2)

    optimal_a_for_each_p = []

    for p_candidate in p_values:
        expected_yields = []
        for a_candidate in ai_values:
            yield_val = compute_expected_yield(
                p_candidate, a_candidate, Delta, h, C1, C2, n_exp, m_exp
            )
            expected_yields.append(yield_val)

        optimal_a = ai_values[np.argmax(expected_yields)]
        optimal_a_for_each_p.append(optimal_a)

    print("Computation complete.")
    return np.array(optimal_a_for_each_p)


# =============================================================================
#  Forward-looking (DP) Solver
# =============================================================================

class GeneticOptimizer:
    """
    Dynamic programming (forward-looking regulator) solver.
    Solves the Bellman equation via value function iteration.
    """

    def __init__(self, h=0.5, Delta=1.0, beta=0.95,
                 C1=1.0, C2=1.0, n=2, m=2, cost_type='quadratic'):
        """
        Initialize the genetic optimization problem.

        Args:
            h: Dominance coefficient
            Delta: Yield advantage coefficient
            beta: Discount factor
            C1, C2: Loss function multipliers
            n, m: Loss function exponents
            cost_type: Cost function type ('quadratic' or 'exponential')
        """
        self.h = h
        self.Delta = Delta
        self.beta = beta
        self.C1 = C1
        self.C2 = C2
        self.n = n
        self.m = m
        self.cost_type = cost_type

        # State space discretization
        self.n_points = 501
        self.p_grid = np.linspace(0, 1, self.n_points)
        self.dp = self.p_grid[1] - self.p_grid[0]

        # Initialize value function
        self.V = np.zeros(self.n_points)
        self.V_new = np.zeros(self.n_points)
        self.policy = np.zeros(self.n_points)

    def cost_function(self, a, p):
        """
        Cost function — delegates to the unified loss_function.

        In quadratic mode:
            Original formula: c * ((a*p)^2) * 2 + c * (((1-a)*(1-p))^2) * 2
            Equivalent to: loss_function(a, p, C1, C2, n, m)
            where C1 and C2 already incorporate the multiplier and coefficient.
        """
        if self.cost_type == 'quadratic':
            return loss_function(a, p, self.C1, self.C2, self.n, self.m)
        elif self.cost_type == 'exponential':
            x = a * p + (1 - a) * (1 - p)
            return self.C1 * (np.exp(x) - 1)
        else:
            raise ValueError("cost_type must be 'quadratic' or 'exponential'")

    def immediate_payoff(self, a, p):
        """Immediate payoff function."""
        heterozygous = a * (1 - p) + (1 - a) * p
        return 1 + (a * p + self.h * heterozygous) * self.Delta - self.cost_function(a, p)

    def state_transition(self, a, p):
        """State transition equation."""
        return 0.5 * (a + p)

    def interpolate_value(self, p):
        """Linear interpolation of the value function."""
        if p <= 0:
            return self.V[0]
        elif p >= 1:
            return self.V[-1]
        else:
            idx = int(p / self.dp)
            if idx >= self.n_points - 1:
                return self.V[-1]
            weight = (p - self.p_grid[idx]) / self.dp
            return self.V[idx] * (1 - weight) + self.V[idx + 1] * weight

    def bellman_operator(self, p_idx):
        """Bellman operator: find optimal control a for a given state p."""
        p = self.p_grid[p_idx]

        a_candidates = np.linspace(0, 1, 501)
        values = []
        for a in a_candidates:
            payoff = self.immediate_payoff(a, p)
            next_p = self.state_transition(a, p)
            future_value = self.interpolate_value(next_p)
            values.append(payoff + self.beta * future_value)

        best_idx = np.argmax(values)
        return a_candidates[best_idx], values[best_idx]

    def value_function_iteration(self, max_iter=5000, tol=1e-6, verbose=True):
        """Value function iteration algorithm."""
        for iteration in range(max_iter):
            for i in range(self.n_points):
                optimal_a, optimal_value = self.bellman_operator(i)
                self.V_new[i] = optimal_value
                self.policy[i] = optimal_a

            max_diff = np.max(np.abs(self.V_new - self.V))

            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Max difference = {max_diff:.6f}")

            if max_diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            self.V = self.V_new.copy()

        return iteration + 1

    def simulate_trajectory(self, p0, T=20):
        """Simulate optimal trajectory."""
        p_trajectory = np.zeros(T + 1)
        a_trajectory = np.zeros(T)
        p_trajectory[0] = p0

        for t in range(T):
            p_current = p_trajectory[t]
            if p_current <= 0:
                a_optimal = self.policy[0]
            elif p_current >= 1:
                a_optimal = self.policy[-1]
            else:
                idx = int(p_current / self.dp)
                if idx >= self.n_points - 1:
                    a_optimal = self.policy[-1]
                else:
                    weight = (p_current - self.p_grid[idx]) / self.dp
                    a_optimal = (self.policy[idx] * (1 - weight)
                                 + self.policy[idx + 1] * weight)

            a_trajectory[t] = a_optimal
            p_trajectory[t + 1] = self.state_transition(a_optimal, p_current)

        return p_trajectory, a_trajectory

    def analyze_steady_state(self):
        """Analyze steady state."""
        print("Steady-state analysis:")
        print("-" * 40)

        steady_states = []
        for i in range(self.n_points):
            p = self.p_grid[i]
            a = self.policy[i]
            if abs(p - a) < 0.01:
                steady_states.append((p, a))

        if steady_states:
            print("Steady states found:")
            for p, a in steady_states:
                print(f"  p* = {p:.3f}, a* = {a:.3f}")
        else:
            print("No obvious steady states found.")

        print(f"\nPolicy function summary:")
        print(f"  Min control: {np.min(self.policy):.3f}")
        print(f"  Max control: {np.max(self.policy):.3f}")
        print(f"  Mean control: {np.mean(self.policy):.3f}")
