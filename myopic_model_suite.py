# myopic_model_suite.py
# 一个集成了“短视调节者”模型分析所有功能的套件
# 功能包括：
# 1. 计算并绘制最优策略函数 a*(p)
# 2. 模拟 p 和 a 在多代中的动态演化
# 3. 可视化解释：为给定的p，绘制预期产量 vs a 的关系曲线，展示 a* 如何被确定

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

# --- 核心模块导入 ---
# 假设这些模块文件与本文件在同一目录下
from module_simulation import calculate_yield_one_generation, run_dynamic_policy_simulation
from module_plot import plot_p_evolution_comparison

# --- 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================================================
# 功能函数 1: 计算最优策略函数 a*(p)
# ==========================================================================
def calculate_policy_function(p_values, ai_values, static_params):
    """
    为每个给定的等位基因频率 p 计算最优的策略 a*。
    """
    print("--- 正在计算最优策略函数 a*(p) ---")
    optimal_a_for_each_p = []
    for p_candidate in p_values:
        expected_yields = []
        for a_candidate in ai_values:
            # 使用现有函数计算给定(p, a)组合下的预期产量
            E_prop_aa = (1 - p_candidate) * (1 - a_candidate)
            E_prop_AA = p_candidate * a_candidate
            E_prop_Aa = 1 - E_prop_aa - E_prop_AA

            yield_val = calculate_yield_one_generation(
                E_prop_aa, E_prop_Aa, E_prop_AA,
                static_params['Delta'], static_params['h'], static_params['pi'], static_params['gamma']
            )
            expected_yields.append(yield_val)

        optimal_a = ai_values[np.argmax(expected_yields)]
        optimal_a_for_each_p.append(optimal_a)
    print("计算完成。")
    return np.array(optimal_a_for_each_p)


# ==========================================================================
# 功能函数 2: 【新增】可视化解释函数
# ==========================================================================
def plot_yield_curves(p_values_to_show, ai_values, static_params, results_dir):
    """
    为给定的几个p值，绘制“预期产量 vs 策略a”的函数图像，并标记最优点。
    这张图直观地解释了最优策略 a* 是如何被找到的。
    """
    print(f"\n--- 正在为 p = {p_values_to_show} 生成产量曲线解释图 ---")
    plt.figure(figsize=(7, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(p_values_to_show)))

    for i, p_val in enumerate(p_values_to_show):
        # 计算当前p值下，所有a对应的预期产量
        expected_yields = []
        for a_candidate in ai_values:
            E_prop_aa = (1 - p_val) * (1 - a_candidate)
            E_prop_AA = p_val * a_candidate
            E_prop_Aa = 1 - E_prop_aa - E_prop_AA

            yield_val = calculate_yield_one_generation(
                E_prop_aa, E_prop_Aa, E_prop_AA,
                static_params['Delta'], static_params['h'], static_params['pi'], static_params['gamma']
            )
            expected_yields.append(yield_val)

        # 找到最优点
        max_yield_idx = np.argmax(expected_yields)
        optimal_a = ai_values[max_yield_idx]
        max_yield = expected_yields[max_yield_idx]

        # 绘制曲线
        plt.plot(ai_values, expected_yields, color=colors[i], label=f'When p = {p_val:.1f}')

        # 标记最高点
        plt.plot(optimal_a, max_yield, 's', color='red', markersize=6,
                 markerfacecolor='none', markeredgecolor='red',
                 label=f'Optimal Choice a*({p_val:.2f}) ≈ {optimal_a:.2f}')

    plt.title('Maximizing Expected Yield of the Current Generation', fontsize=16)
    plt.xlabel('Control Variable: a (Allele G in Controlled Breeding)', fontsize=12)
    plt.ylabel('Expected Yield of the Current Generation', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    output_filename = "yield_curve_explanation.png"
    output_filepath = os.path.join(results_dir, output_filename)
    plt.savefig(output_filepath, dpi=300)
    print(f"产量曲线解释图已保存到: {output_filepath}")
    plt.show()


# ==========================================================================
# 主程序入口：模型“控制面板”
# ==========================================================================
if __name__ == '__main__':
    # --- 1. 控制面板：选择要运行的分析任务 ---
    # 将你想要运行的任务设置为 True
    RUN_POLICY_FUNCTION_ANALYSIS = True
    RUN_DYNAMIC_SIMULATION = True
    RUN_YIELD_CURVE_EXPLANATION = True

    # --- 2. 定义结果文件夹和共享参数 ---
    RESULTS_DIR = "myopic_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 在这里统一设置参数，所有分析都将使用这套参数
    MODEL_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
        "pi": 1.0,
        "gamma": 0.8,
        "population_size": 100000,
        "initial_A_proportion": 0.6,
    }

    # 定义 p 和 a 的分析范围
    p_values_grid = np.linspace(0, 1, 501)
    a_values_grid = np.linspace(0, 1, 501)

    # --- 3. 根据“控制面板”的设置，执行相应的任务 ---

    # 任务一：分析和绘制最优策略函数 a*(p)
    if RUN_POLICY_FUNCTION_ANALYSIS:
        print("\n" + "=" * 50)
        print("任务1: 分析最优策略函数 a*(p)")
        print("=" * 50)
        optimal_policy = calculate_policy_function(
            p_values=p_values_grid,
            ai_values=a_values_grid,
            static_params=MODEL_PARAMS
        )

        plt.figure(figsize=(7, 7))
        plt.plot(p_values_grid, optimal_policy, 'b-', linewidth=2, label='最优策略 a*(p)')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='a = p (无干预)')
        plt.title(
            f'短视最优策略函数 a*(p)\n(h={MODEL_PARAMS["h"]}, Δ={MODEL_PARAMS["Delta"]}, π={MODEL_PARAMS["pi"]}, γ={MODEL_PARAMS["gamma"]})')
        plt.xlabel("p (等位基因 A 的频率)")
        plt.ylabel("a* (最优策略)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "policy_function_a_vs_p.png"), dpi=300)
        plt.show()

    # 任务二：运行动态模拟
    if RUN_DYNAMIC_SIMULATION:
        print("\n" + "=" * 50)
        print("任务2: 运行多初始值下的动态演化模拟")
        print("=" * 50)

        # 【修改这里】定义要测试的多个初始 p 值
        initial_p_values_to_test = [0.2, 0.5, 0.8]
        total_generations = 10
        all_simulation_results = []

        print(f"将为初始 p = {initial_p_values_to_test} 分别运行模拟...")

        # 【修改这里】循环遍历每个初始p值
        for p0 in initial_p_values_to_test:
            print(f"  - 正在运行 p0 = {p0:.2f} 的模拟...")

            # 每次循环都为当前模拟创建独立的参数字典
            current_sim_params = MODEL_PARAMS.copy()
            current_sim_params['initial_A_proportion'] = p0  # 模拟函数使用这个键作为p0

            dynamic_results = run_dynamic_policy_simulation(
                total_generations=total_generations,
                ai_values=a_values_grid,
                static_params=current_sim_params
            )
            all_simulation_results.append(dynamic_results)

        print("\n所有模拟运行完成。")

        # 【修改这里】调用新的、专门用于对比的绘图函数
        plot_p_evolution_comparison(
            results_list=all_simulation_results,
            initial_p_list=initial_p_values_to_test,
            static_params=MODEL_PARAMS,  # 传入共享参数用于生成标题
            output_dir=RESULTS_DIR
        )

        # 注意：保存结果的逻辑也需要调整。这里我们不再保存，因为绘图是主要目的。
        # 如果需要保存，可以取消下面的注释
        print("正在分别保存每个模拟的结果...")
        for i, p0 in enumerate(initial_p_values_to_test):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"dynamic_sim_p0_{p0:.2f}_{timestamp}.npy"
            np.save(os.path.join(RESULTS_DIR, filename), all_simulation_results[i])


    # 任务三：绘制产量曲线解释图
    if RUN_YIELD_CURVE_EXPLANATION:
        print("\n" + "=" * 50)
        print("任务3: 生成策略函数的可视化解释图")
        print("=" * 50)
        p_values_for_explanation = [0.3, 0.5, 0.8]  # 选择你希望展示的p值
        plot_yield_curves(
            p_values_to_show=p_values_for_explanation,
            ai_values=a_values_grid,
            static_params=MODEL_PARAMS,
            results_dir=RESULTS_DIR
        )

    print("\n所有选定任务已完成。")