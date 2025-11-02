import numpy as np
import time

# =============================================================================
#  HELPER FUNCTIONS (Core Genetic Logic)
# =============================================================================

def initialize_population(pop_size, initial_A_proportion):
    """
    初始化种群，用 1 代表 'A', 0 代表 'a'。

    Args:
        pop_size (int): 种群大小 (N).
        initial_A_proportion (float): 'A'等位基因的初始频率。

    Returns:
        np.ndarray: 一个 (pop_size, 2) 的整数数组，代表种群基因型。
    """
    num_alleles = 2 * pop_size
    num_A_alleles = int(num_alleles * initial_A_proportion)
    
    # 创建包含所有等位基因的一维数组
    alleles = np.zeros(num_alleles, dtype=np.int8)
    alleles[:num_A_alleles] = 1
    
    # 随机打乱并重塑为 N x 2 的矩阵
    np.random.shuffle(alleles)
    return alleles.reshape((pop_size, 2))

def calculate_genotype_proportions(population):
    """
    高效计算基因型频率。

    Args:
        population (np.ndarray): 当前种群基因型矩阵。

    Returns:
        tuple: 包含 (aa, Aa, AA) 基因型比例的元组。
    """
    pop_size = population.shape[0]
    # 对每个个体的等位基因求和 (A=1, a=0)
    # aa -> sum=0, Aa -> sum=1, AA -> sum=2
    allele_sums = np.sum(population, axis=1)
    
    aa_prop = np.sum(allele_sums == 0) / pop_size
    Aa_prop = np.sum(allele_sums == 1) / pop_size
    AA_prop = np.sum(allele_sums == 2) / pop_size
    
    return aa_prop, Aa_prop, AA_prop

def calculate_yield_one_generation(aa_prop, Aa_prop, AA_prop, Delta, h, pi, gamma):
    """
    根据基因型频率计算当前代的产出。

    Args:
        aa_prop, Aa_prop, AA_prop (float): 基因型频率。
        Delta, h, pi, gamma (float): 模型参数。

    Returns:
        float: 当前代的总产出。
    """
    # 1. 根据新模型计算本代的有效疾病爆发概率
    # 注意：输入的'pi'现在是系数，而不是概率
    effective_pi = pi * (AA_prop + aa_prop)

    # 2. 【重要】确保有效概率不会超过1，增加模型稳健性
    effective_pi = min(effective_pi, 1.0)

    # # 3. 计算疾病对产出的影响因子
    # disease_effect = 1 - effective_pi # * gamma
    #
    # # 4. 计算各基因型的最终产出并加总
    # aa_yield = aa_prop * 1.0 * disease_effect
    # Aa_yield = Aa_prop * (1.0 + h * Delta)  # 杂合子不受疾病影响
    # AA_yield = AA_prop * (1.0 + Delta) * disease_effect

    loss =   (AA_prop ** 2) * 2 + (aa_prop ** 2) * 2
    # loss = gamma * effective_pi ** 2
    aa_yield = aa_prop * 1.0
    Aa_yield = Aa_prop * (1.0 + h * Delta)  # 杂合子不受疾病影响
    AA_yield = AA_prop * (1.0 + Delta)

    return aa_yield + Aa_yield + AA_yield - loss


# # 这是老版本的calculate_yield_one_generation
# def calculate_yield_one_generation(aa_prop, Aa_prop, AA_prop, Delta, h, pi, gamma):
#     """
#     根据基因型频率计算当前代的产出。
#
#     Args:
#         aa_prop, Aa_prop, AA_prop (float): 基因型频率。
#         Delta, h, pi, gamma (float): 模型参数。
#
#     Returns:
#         float: 当前代的总产出。
#     """
#     disease_effect = 1 - pi * gamma
#
#     # 基础产出 * 疾病影响 * 频率
#     aa_yield = aa_prop * 1.0 * disease_effect
#     Aa_yield = Aa_prop * (1.0 + h * Delta)  # 杂合子不受疾病影响
#     AA_yield = AA_prop * (1.0 + Delta) * disease_effect
#
#     return aa_yield + Aa_yield + AA_yield


def create_offspring(parents, pop_size, ai_proportion):
    """
    根据“随机母本 + 人工授精父本”规则创建后代。
    这是一个完全向量化的版本，效率极高。

    Args:
        parents (np.ndarray): 父代种群矩阵。
        pop_size (int): 种群大小。
        ai_proportion (float): 人工授精中'A'基因的比例。

    Returns:
        np.ndarray: 新一代的种群矩阵。
    """
    # 1. 从父代中随机选择 N 个母本（允许重复选择）
    mother_indices = np.random.randint(0, pop_size, size=pop_size)
    mothers = parents[mother_indices, :]
    
    # 2. 从每个母本中随机选择一个等位基因，形成母方配子
    maternal_gamete_indices = np.random.randint(0, 2, size=pop_size)
    maternal_gametes = mothers[np.arange(pop_size), maternal_gamete_indices]
    
    # 3. 根据 ai_proportion 生成父方配子
    paternal_gametes = np.random.choice([1, 0], size=pop_size, p=[ai_proportion, 1 - ai_proportion])
    
    # 4. 组合成新的后代
    return np.column_stack((maternal_gametes, paternal_gametes))


# =============================================================================
#  LEVEL 1: THE CORE SIMULATION ENGINE (新函数)
# =============================================================================
def run_single_simulation(population_size, num_generations, initial_A_proportion,
                          ai_proportion, Delta, h, pi, gamma):
    """
    核心引擎：为一组固定的参数运行一次完整的模拟，并返回平均产出。

    Args:
        所有参数都为单个数值 (int or float)。

    Returns:
        float: 这次模拟的平均产出。
    """
    population = initialize_population(population_size, initial_A_proportion)
    total_yield = 0
    for _ in range(num_generations):
        aa_prop, Aa_prop, AA_prop = calculate_genotype_proportions(population)
        gen_yield = calculate_yield_one_generation(aa_prop, Aa_prop, AA_prop, Delta, h, pi, gamma)
        total_yield += gen_yield
        population = create_offspring(population, population_size, ai_proportion)

    return total_yield / num_generations


# =============================================================================
#  LEVEL 2: THE GENERAL COMPARATIVE STATICS FUNCTION (新函数)
# =============================================================================
def perform_comparative_statics(static_params, sweep_param_name, sweep_param_values, ai_values):
    """
    通用的比较静态分析函数。

    Args:
        static_params (dict): 在本次分析中保持不变的参数字典。
        sweep_param_name (str): 需要进行扫描的参数的名称 (e.g., 'h', 'Delta')。
        sweep_param_values (np.ndarray): 被扫描参数的取值范围。
        ai_values (np.ndarray): 需要优化的 'a' 的取值范围。

    Returns:
        dict: 包含所有分析结果的字典。
    """
    print(f"开始对参数 '{sweep_param_name}' 进行比较静态分析...")
    start_time = time.time()

    num_sweep = len(sweep_param_values)
    avg_yields_mesh = np.zeros((num_sweep, len(ai_values)))
    potential_benefits = np.zeros(num_sweep)
    optimal_ai_proportions = np.zeros(num_sweep)

    # 动态参数字典，用于每次调用核心引擎
    dynamic_params = static_params.copy()

    for i, sweep_val in enumerate(sweep_param_values):
        # 更新当前要扫描的参数值
        dynamic_params[sweep_param_name] = sweep_val

        avg_yields_for_current_sweep = np.zeros(len(ai_values))

        # 内层循环：对于当前的sweep_val，找到最优的 a
        for j, ai_prop in enumerate(ai_values):
            dynamic_params['ai_proportion'] = ai_prop

            # 调用核心引擎！
            avg_yields_for_current_sweep[j] = run_single_simulation(**dynamic_params)

        avg_yields_mesh[i, :] = avg_yields_for_current_sweep

        max_yield = np.max(avg_yields_for_current_sweep)
        max_index = np.argmax(avg_yields_for_current_sweep)
        yield_at_ai_1 = avg_yields_for_current_sweep[-1]

        potential_benefits[i] = max_yield - yield_at_ai_1
        optimal_ai_proportions[i] = ai_values[max_index]

        print(f"  {sweep_param_name} = {sweep_val:.3f} 完成... (进度: {i + 1}/{num_sweep})")

    end_time = time.time()
    print(f"分析完成！总耗时: {end_time - start_time:.2f} 秒。")

    results = {
        "sweep_param_name": sweep_param_name,
        "sweep_param_values": sweep_param_values,
        "ai_values": ai_values,
        "avg_yields_mesh": avg_yields_mesh,
        "potential_benefits": potential_benefits,
        "optimal_ai_proportions": optimal_ai_proportions,
        "static_params": static_params
    }
    return results


# =============================================================================
#  LEVEL 3: DYNAMIC SIMULATION
# =============================================================================

def run_dynamic_policy_simulation(total_generations, ai_values, static_params):
    """
    运行动态策略模拟，在每一代都寻找最优的'a'。

    Args:
        total_generations (int): 总共要模拟的代数。
        ai_values (np.ndarray): 每一代中供选择的候选'a'值的范围。
        static_params (dict): 包含所有固定的生物学/经济学参数的字典。

    Returns:
        dict: 包含所有时间序列结果的字典。
    """
    print("--- 开始动态策略模拟 ---")
    start_time = time.time()

    # 从参数字典中解包
    pop_size = static_params['population_size']
    initial_A_prop = static_params['initial_A_proportion']
    h, Delta, pi, gamma = static_params['h'], static_params['Delta'], static_params['pi'], static_params['gamma']

    # 初始化
    current_population = initialize_population(pop_size, initial_A_prop)

    # 创建列表存储时间序列数据
    optimal_a_series = []
    genotype_proportions_series = []
    allele_p_series = []  # 'p' 指的是等位基因 'a' 的频率

    # --- 主循环 ---
    for gen in range(total_generations):
        # 1. 分析当前种群状态
        prop_aa, prop_Aa, prop_AA = calculate_genotype_proportions(current_population)
        p_a_maternal = prop_aa + 0.5 * prop_Aa  # 计算当前母本池中 'a' 的频率
        p_A_maternal = 1 - p_a_maternal

        # 2. 寻找当前代的最优 a_t
        expected_yields = []
        for a_candidate in ai_values:
            # 解析计算使用该 a_candidate 后，下一代的预期基因型频率
            p_a_paternal = 1 - a_candidate
            p_A_paternal = a_candidate

            E_prop_aa_next = p_a_maternal * p_a_paternal
            E_prop_AA_next = p_A_maternal * p_A_paternal
            E_prop_Aa_next = 1 - E_prop_aa_next - E_prop_AA_next

            # 根据预期的基因型频率计算预期产出
            yield_val = calculate_yield_one_generation(
                E_prop_aa_next, E_prop_Aa_next, E_prop_AA_next, Delta, h, pi, gamma
            )
            expected_yields.append(yield_val)

        # 找到最大化预期产出的 a
        optimal_a_t = ai_values[np.argmax(expected_yields)]

        # 3. 记录本期数据
        optimal_a_series.append(optimal_a_t)
        genotype_proportions_series.append((prop_aa, prop_Aa, prop_AA))
        allele_p_series.append(p_A_maternal)

        # 4. 使用本期找到的最优 a_t 来生成下一代种群
        current_population = create_offspring(current_population, pop_size, optimal_a_t)

        if (gen + 1) % 10 == 0:  # 每10代打印一次进度
            print(f"  Generation {gen + 1}/{total_generations} 完成...")

    end_time = time.time()
    print(f"动态模拟完成！总耗时: {end_time - start_time:.2f} 秒。")

    return {
        "total_generations": total_generations,
        "optimal_a_series": np.array(optimal_a_series),
        "genotype_proportions_series": np.array(genotype_proportions_series),
        "allele_p_series": np.array(allele_p_series)
    }