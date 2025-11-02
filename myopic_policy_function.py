# analyze_policy_function.py
# 一个专门用于分析和绘制最优策略函数 a*(p) 的脚本。

import numpy as np
import os
import matplotlib.pyplot as plt

# 导入我们模型的核心“积木”
# 注意：我们只需要产出计算函数，因为这里没有实际的种群演化
from module_simulation import calculate_yield_one_generation


def calculate_policy_function(p_values, ai_values, static_params):
    """
    为每个给定的等位基因频率 p 计算最优的策略 a*。

    Args:
        p_values (np.ndarray): 'a'等位基因频率的取值范围。
        ai_values (np.ndarray): 候选 'a' 策略的取值范围。
        static_params (dict): 包含所有固定参数的字典。

    Returns:
        np.ndarray: 每个 p 值对应的最优 a* 值。
    """
    print("--- 开始计算最优策略函数 a*(p) ---")

    # 从参数字典中解包
    h, Delta, pi, gamma = static_params['h'], static_params['Delta'], static_params['pi'], static_params['gamma']

    optimal_a_for_each_p = []

    for p_candidate in p_values:
        # p_candidate 是当前母本池中 'a' 的频率 (p_a_maternal)
        p_A_maternal = p_candidate
        p_a_maternal = 1 - p_A_maternal

        expected_yields = []
        # 对于给定的p，测试所有可能的a，找到最好的一个
        for a_candidate in ai_values:
            p_a_paternal = 1 - a_candidate
            p_A_paternal = a_candidate

            # 计算下一代的预期基因型频率
            E_prop_aa_next = p_a_maternal * p_a_paternal
            E_prop_AA_next = p_A_maternal * p_A_paternal
            E_prop_Aa_next = 1 - E_prop_aa_next - E_prop_AA_next

            # 计算此预期状态下的产出
            yield_val = calculate_yield_one_generation(
                E_prop_aa_next, E_prop_Aa_next, E_prop_AA_next, Delta, h, pi, gamma
            )
            expected_yields.append(yield_val)

        # 找到使预期产出最大化的 a
        optimal_a = ai_values[np.argmax(expected_yields)]
        optimal_a_for_each_p.append(optimal_a)

    print("计算完成。")
    return np.array(optimal_a_for_each_p)


if __name__ == '__main__':
    # --- 1. 定义结果文件夹并确保其存在 ---
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. 设置模型参数 ---
    # 您可以在这里调整参数，观察策略函数如何变化
    MODEL_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
        "pi": 0.5,  # 别忘了这里的pi是系数
        "gamma": 1.0
    }

    # --- 3. 定义 p 和 a 的分析范围 ---
    p_values_to_scan = np.linspace(0, 1, 501)  # 用更高的分辨率扫描 p
    a_values_to_test = np.linspace(0, 1, 501)

    # --- 4. 运行计算 ---
    optimal_as = calculate_policy_function(
        p_values=p_values_to_scan,
        ai_values=a_values_to_test,
        static_params=MODEL_PARAMS
    )

    # --- 5. 绘图 ---
    print("开始绘图...")
    plt.figure(figsize=(7, 7))
    plt.plot(p_values_to_scan, optimal_as, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x')
    plt.title(
        f'Optimal Policy Function: a*(p)\n(h={MODEL_PARAMS["h"]}, Δ={MODEL_PARAMS["Delta"]}, π_coeff={MODEL_PARAMS["pi"]})')
    plt.xlabel("Allele 'a' Frequency (p)")
    plt.ylabel("Optimal Policy (a*)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    output_filename = "policy_function_a_vs_p.png"
    output_filepath = os.path.join(RESULTS_DIR, output_filename)
    plt.savefig(output_filepath, dpi=300)
    print(f"策略函数图像已保存到: {output_filepath}")

    plt.show()