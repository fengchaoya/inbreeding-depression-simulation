# compare_policies.py
# 一个用于比较 Myopic Regulator 和 Dynamic Regulator 策略函数的脚本

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 从你的现有脚本中导入核心功能 ---
# 从 myopic 脚本中导入策略计算函数
from myopic_model_suite import calculate_policy_function
# 从 dp 脚本中导入优化器类
from dp_simulation import GeneticOptimizer

# --- 可选：解决中文显示问题的代码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_comparison():
    """
    运行两种策略的计算并绘制对比图。
    """
    print("--- 开始对比 Myopic 和 Dynamic 策略 ---")

    # --- 2. 定义共享的模型参数 ---
    # 在这里统一设置参数，确保两种模型在相同的条件下进行比较
    SHARED_PARAMS = {
        "h": 0.6,
        "Delta": 0.4,
    }

    # Myopic Regulator 特有的参数 (pi, gamma)
    MYOPIC_EXTRA_PARAMS = {
        "pi": 0.5,
        "gamma": 1.0
    }

    # Dynamic Regulator 特有的参数 (beta, c)
    DP_EXTRA_PARAMS = {
        "beta": 0.95,
        "c": 1.0,
        "cost_type": 'quadratic'
    }

    # 定义分析用的 p 值网格
    p_grid = np.linspace(0, 1, 501)

    # --- 3. 计算 Myopic Regulator 的策略函数 ---
    print("\n[1/2] 正在计算 Myopic Policy Function...")
    # 组合 myopic 模型所需的所有参数
    myopic_params_full = {**SHARED_PARAMS, **MYOPIC_EXTRA_PARAMS}

    # a_values_to_test 应该和 p_grid 保持一致以获得最佳匹配
    a_grid_myopic = np.linspace(0, 1, 501)

    policy_myopic = calculate_policy_function(
        p_values=p_grid,
        ai_values=a_grid_myopic,
        static_params=myopic_params_full
    )
    print("Myopic Policy 计算完成。")

    # --- 4. 计算 Dynamic Regulator 的策略函数 ---
    print("\n[2/2] 正在计算 Dynamic Policy Function...")
    optimizer = GeneticOptimizer(
        h=SHARED_PARAMS['h'],
        Delta=SHARED_PARAMS['Delta'],
        beta=DP_EXTRA_PARAMS['beta'],
        c=DP_EXTRA_PARAMS['c'],
        cost_type=DP_EXTRA_PARAMS['cost_type']
    )

    # 运行值函数迭代来求解模型，设置 verbose=False 来保持输出整洁
    optimizer.value_function_iteration(verbose=False, tol=1e-6)
    policy_dynamic = optimizer.policy
    # DP模型的p网格是从optimizer对象中获取的，以确保一致性
    p_grid_dynamic = optimizer.p_grid
    print("Dynamic Policy 计算完成。")

    # --- 5. 绘制对比图 ---
    print("\n开始绘图...")
    plt.figure(figsize=(8, 8))

    # 绘制 Myopic Policy
    plt.plot(p_grid, policy_myopic, 'b-', linewidth=2.5, label='Myopic Regulator')

    # 绘制 Dynamic Policy
    plt.plot(p_grid_dynamic, policy_dynamic, 'r--', linewidth=2.5, label=' Forward-looking Regulator')

    # 绘制 y=x 参考线
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='a = p ')

    # 设置图表标题和标签
    title_str = (f' Comparison of Policy Functions\n'
                 f'Shared parameters: h={SHARED_PARAMS["h"]}, Δ={SHARED_PARAMS["Delta"]}\n'
                 f'Dynamic parameter: β={DP_EXTRA_PARAMS["beta"]}')
    plt.title(title_str, fontsize=14)
    plt.xlabel("p (State Variable)", fontsize=12)
    plt.ylabel("a (Control Variable)", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.gca().set_aspect('equal', adjustable='box')  # 使x,y轴等长，y=x更好看
    plt.tight_layout()

    # 保存图像
    os.makedirs("comparison_results", exist_ok=True)
    output_path = os.path.join("comparison_results", "policy_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"对比图像已保存到: {output_path}")

    plt.show()


if __name__ == '__main__':
    run_comparison()