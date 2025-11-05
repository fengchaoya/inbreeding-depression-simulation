# Simulation Strategy and Computational Methods

This document details the computational framework and algorithms used to simulate the genetic and yield dynamics of a population under different regulatory strategies. The focus is on the specific implementation details of the simulation engine, as reflected in the project's Python code.

---

## 1. Core Simulation Mechanics

The simulation is built around a discrete-time, agent-based model where the state of the system evolves in generational steps. The following mechanics are shared across both the myopic and forward-looking models.

### 1.1. Population Representation

*   An individual's genotype is represented by two alleles. The allele 'A' is encoded as `1` and allele 'a' is encoded as `0`.
*   The entire population is stored in a NumPy array of shape `(N, 2)`, where `N` is the population size. Each row represents an individual, and the two columns represent its pair of alleles.
*   The simulation is initialized using the `initialize_population` function, which creates this array based on a specified initial frequency for allele 'A'.

### 1.2. The Generational Cycle: From Parents to Offspring

The simulation proceeds in a generational loop. The creation of a new generation from a parent generation is a two-step process implemented in the `create_offspring` function, which models the controlled breeding strategy.

1.  **Formation of Maternal Gametes:**
    *   First, `N` mothers are randomly selected *with replacement* from the parent population.
    *   Each selected mother contributes exactly one of her two alleles, chosen at random, to form a "maternal gamete". This creates a pool of `N` alleles that represents the genetic contribution from the unmanaged, female side of the population.

2.  **Generation of Paternal Gametes (The Control Step):**
    *   The paternal contribution is entirely determined by the regulator's control variable, `a`. This variable represents the desired frequency of allele 'A' in the artificial insemination pool.
    *   A pool of `N` "paternal gametes" is generated synthetically based on this frequency. For each of the `N` offspring to be created, a paternal allele is drawn from a Bernoulli distribution where the probability of drawing allele 'A' (`1`) is exactly `a`, and the probability of drawing allele 'a' (`0`) is `1-a`.

3.  **Formation of the Next Generation:**
    *   The `N` maternal gametes and `N` paternal gametes are paired one-to-one to form the `N` individuals of the next generation, completing the cycle.

### 1.3. Yield Calculation in a Single Generation

The total yield for any given generation is calculated by the `calculate_yield_one_generation` function based on the proportions of the three possible genotypes (`aa`, `Aa`, `AA`) in that generation. The calculation follows these steps:

1.  **Base Yield Calculation:** The foundational yield is determined by the genetic potential of each genotype. The total base yield is the sum of the contributions from each group:
    `Yield_base = (prop_aa * 1.0) + (prop_Aa * (1.0 + h*Δ)) + (prop_AA * (1.0 + Δ))`
    where `h` is the dominance coefficient and `Δ` is the selection advantage of the 'AA' genotype.

2.  **Cost/Loss Calculation (Inbreeding Depression Effect):** The model incorporates a cost associated with a lack of genetic diversity. This is implemented as a loss function that penalizes high proportions of homozygotes (`aa` and `AA`). The specific loss is calculated as:
    `Loss = c * ((prop_AA^2) * 2 + (prop_aa^2) * 2)`
    where `c` is a cost coefficient.

3.  **Final Yield:** The net yield for the generation is the base yield minus the loss:
    `Yield_final = Yield_base - Loss`

---

## 2. Model 1: The Myopic Regulator

The myopic regulator's goal is to select the control `a` that maximizes the expected yield of the *very next generation*.

### Simulation Strategy

The optimal policy `a*(p)` is found numerically using a grid-search approach, as implemented in `myopic_model_suite.py`.

1.  **Grid Setup:** The state space `p` (frequency of allele 'A') and the action space `a` are both discretized into fine-grained, uniform grids (e.g., 501 points from 0 to 1).
2.  **Iterative Search:** The algorithm iterates through each possible state `p_i` on the grid.
3.  **Expected Yield Evaluation:** For a given `p_i`, the algorithm then iterates through every possible action `a_j` on its grid. For each `(p_i, a_j)` pair, it analytically calculates the *expected* genotype proportions of the next generation. These expected proportions are then fed into the `calculate_yield_one_generation` function to compute a single expected yield value.
4.  **Policy Determination:** The action `a_j` that results in the highest expected yield is identified and stored as the optimal policy `a*(p_i)` for that state. This process is repeated for all `p_i` to map out the entire policy function.

---

## 3. Model 2: The Forward-Looking Regulator

The forward-looking regulator's goal is to find a policy `a*(p)` that maximizes the total discounted sum of all future yields. This is solved using Value Function Iteration (VFI).

### Simulation Strategy: Value Function Iteration

The VFI algorithm, implemented in the `value_function_iteration` method of the `GeneticOptimizer` class, numerically solves the Bellman equation. It works as follows:

1.  **Initialization:**
    *   The state space `p` is discretized into a uniform grid (501 points).
    *   An initial guess for the value function, `V^0`, is created as a vector of zeros, with one value for each point on the state grid.

2.  **The Iteration Loop:** The algorithm iterates to continuously refine the value function until it converges. Each iteration `k` consists of the following steps:
    *   A new, empty value function vector, `V^{k+1}`, is created.
    *   The algorithm loops through every state `p_i` on the grid.
    *   For each `p_i`, it performs a **maximization step**:
        *   It searches over a fine-grained grid of possible actions `a_j`.
        *   For each action `a_j`, it calculates the total expected value: `Value(a_j) = R(a_j, p_i) + β * V^k(p')`.
            *   `R(a_j, p_i)` is the immediate yield (payoff) calculated using the methods described in Section 1.3.
            *   `p' = 0.5 * (a_j + p_i)` is the state in the next period.
            *   Since `p'` may not fall on a grid point, the value `V^k(p')` is estimated via **linear interpolation** using the values from the *previous* iteration's value function, `V^k`, at the two grid points surrounding `p'`.
    *   The maximum value found among all `a_j` becomes the new, updated value for the current state: `V^{k+1}(p_i)`.

3.  **Convergence Check:** After updating the value for all states `p_i`, the algorithm compares the new value function `V^{k+1}` with the old one `V^k`. It calculates the maximum absolute difference between them (the sup-norm).
    *   If this difference is smaller than a predefined tolerance (e.g., `1e-7`), the value function is considered to have converged, and the loop terminates.
    *   If not, the algorithm sets `V^k = V^{k+1}` and begins the next iteration.

4.  **Final Policy Extraction:** Once converged, the optimal policy `a*(p)` is stored. For each state `p_i`, the optimal action is the `a_j` that yielded the maximum value in the final iteration.

## 4. Model 3: The Robust (Fixed-Horizon) Policy

This model evaluates a simpler, more practical regulatory strategy. It assumes the regulator chooses a single, constant control policy `a` that will be applied for a fixed number of generations `T`. The goal is to find the single value `a_robust*` that maximizes the **average yield** over this entire `T`-generation horizon.

This simulation is the core engine used by the comparative statics analysis in `main.py`, specifically within the `run_single_simulation` and `perform_comparative_statics` functions.

### Simulation Strategy and Optimization

The optimal robust policy is found by simulating the outcome of every possible fixed policy and selecting the best one. The algorithm proceeds as follows:

1.  **Grid Setup:** The action space `a` is discretized into a fine-grained, uniform grid (e.g., 51 points from 0.5 to 1), representing all possible fixed policies the regulator could choose.

2.  **The Outer Loop (Searching for the Optimal `a`):** The algorithm iterates through each candidate policy `a_j` from the action grid. For each candidate, it runs a full, multi-generational simulation to evaluate its long-term performance.

3.  **The Inner Loop (Simulating the `T`-Generation Horizon):** For each candidate policy `a_j`, a complete simulation is run for a fixed number of generations `T` (e.g., `num_generations = 25`).
    *   The simulation starts from a defined initial population state (`initial_A_proportion`).
    *   In every single generation within this `T`-generation loop, the **exact same** control policy `a_j` is applied.
    *   The population evolves according to the core mechanics described in Section 1: maternal gametes are drawn from the current population, while paternal gametes are synthetically generated based on the constant policy `a_j`.
    *   The net yield for each of the `T` generations is calculated and recorded.

4.  **Performance Evaluation:** After the `T` generations are complete, the **average yield** over the entire horizon is calculated. This single value represents the overall performance of the candidate policy `a_j`.

5.  **Policy Determination:** The algorithm compares the average yield produced by each candidate `a_j`. The policy `a_j` that results in the highest overall average yield is selected as the optimal robust policy, `a_robust*`. The comparative statics analysis in `main.py` then explores how this `a_robust*` changes as other underlying model parameters (like `h` or `Δ`) are varied.