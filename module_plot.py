import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def create_plots(results, output_dir):
    """
    根据分析结果生成所有需要的图表 (完全通用的最终版本)。
    """
    print(f"\n开始生成图表，将保存到 '{output_dir}' 文件夹...")

    # 从结果字典中解包数据
    sweep_param_name = results["sweep_param_name"]
    sweep_param_values = results["sweep_param_values"]
    ai_values = results["ai_values"]
    potential_benefits = results["potential_benefits"]
    optimal_ai_proportions = results["optimal_ai_proportions"]
    # 3D图和等高线图需要原始的产出网格数据
    avg_yields_mesh = results["avg_yields_mesh"]

    # --- 设置X轴标签 (这部分保持不变) ---
    xlabel_map = {
        'h': 'Dominance Coefficient (h)',
        'Delta': 'Selection Coefficient (Δ)',
        'pi': 'Disease Incidence (π)',
        'gamma': 'Disease Severity (γ)',
        'num_generations': 'Number of Generations',
        'initial_a_proportion': "Initial 'A' Allele Proportion"
    }
    # Y_label用于3D图的Y轴
    Y_label = xlabel_map.get(sweep_param_name, sweep_param_name)

    # --- 图1和图2: 2D总结图 (这部分保持不变) ---
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_param_values, potential_benefits, 'b-', linewidth=2)
    plt.xlabel(Y_label)  # 使用我们刚定义的标签
    plt.ylabel('Potential Benefit')
    plt.title(f'Potential Benefit vs. {sweep_param_name}')
    plt.grid(True)
    plt.tight_layout()
    filename_benefit = f"benefit_vs_{sweep_param_name}.png"
    filepath_benefit = os.path.join(output_dir, filename_benefit)
    plt.savefig(filepath_benefit, dpi=300)

    plt.figure(figsize=(8, 6))
    plt.plot(sweep_param_values, optimal_ai_proportions, 'b-', linewidth=2)
    plt.xlabel(Y_label)  # 使用我们刚定义的标签
    plt.ylabel('Optimal AI Proportion (a)')
    plt.title(f'Optimal AI Proportion (a) vs. {sweep_param_name}')
    plt.ylim([min(ai_values), max(ai_values)])
    plt.grid(True)
    plt.tight_layout()
    filename_optimal_a = f"optimal_a_vs_{sweep_param_name}.png"
    filepath_optimal_a = os.path.join(output_dir, filename_optimal_a)
    plt.savefig(filepath_optimal_a, dpi=300)

    # --- 以下是针对3D图和等高线图的重大修改 ---

    # 修改1: 使用通用变量生成网格数据
    # X轴是 'a' (ai_values), Y轴是正在扫描的参数 (sweep_param_values)
    X_MESH, Y_MESH = np.meshgrid(ai_values, sweep_param_values)

    # 修改2: 绘制通用的3D曲面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 注意这里Y_MESH和avg_yields_mesh的对应关系
    surf = ax.plot_surface(X_MESH, Y_MESH, avg_yields_mesh, cmap='viridis', edgecolor='none')

    # 修改3: 使用动态的标签和标题
    ax.set_xlabel('AI Proportion (a)')
    ax.set_ylabel(Y_label)  # Y轴标签是动态的
    ax.set_zlabel('Average Yield')
    ax.set_title(f'Average Yield vs. a and {sweep_param_name}')  # 标题是动态的
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Average Yield')
    plt.tight_layout()

    # 修改4: 使用动态的文件名
    filename_surface = f"yield_surface_vs_{sweep_param_name}.png"
    filepath_surface = os.path.join(output_dir, filename_surface)
    plt.savefig(filepath_surface, dpi=300)

    # 修改5: 绘制通用的等高线图 (同样应用所有修改)
    plt.figure(figsize=(9, 7))
    contour = plt.contourf(X_MESH, Y_MESH, avg_yields_mesh, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Average Yield')
    plt.xlabel('AI Proportion (a)')
    plt.ylabel(Y_label)  # Y轴标签是动态的
    plt.title(f'Contour Plot: Yield vs. a and {sweep_param_name}')  # 标题是动态的
    plt.tight_layout()
    filename_contour = f"yield_contour_vs_{sweep_param_name}.png"
    filepath_contour = os.path.join(output_dir, filename_contour)
    plt.savefig(filepath_contour, dpi=300)

    print("所有图表已生成并保存为PNG文件。")
    plt.show()


# module_plotting.py (添加以下新函数)

def plot_p_evolution_comparison(results_list, initial_p_list, static_params, output_dir):
    """
    为多个动态策略模拟的结果绘图，对比不同初始p值下的演化路径。
    """
    print(f"\n开始生成多初始值动态模拟对比图，将保存到 '{output_dir}' 文件夹...")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results_list)))

    if not results_list:
        print("警告：没有提供模拟结果用于绘图。")
        return

    total_generations = 0
    for i, results in enumerate(results_list):
        total_generations = results['total_generations']
        initial_p = initial_p_list[i]

        allele_p_series = results['allele_p_series']
        # 【新增】从结果中提取 a 的时间序列
        optimal_a_series = results['optimal_a_series']

        # X轴：代数。长度是 total_generations
        generations_axis = np.arange(1, total_generations + 1)

        # 此时 generations_axis 和 allele_p_series 长度一致
        plt.plot(generations_axis, allele_p_series, color=colors[i], marker='o',
                 markersize=4, linestyle='-', linewidth=2, label=f'Initial p = {initial_p:.1f}')

        # 【新增】绘制 a 的曲线，使用相同颜色但不同线型
        plt.plot(generations_axis, optimal_a_series, color=colors[i], marker='x',
                 markersize=5, linestyle='--', linewidth=2, label=f'a (Initial p={initial_p:.1f})')

    # ... (函数的其余部分保持不变) ...
    h = static_params.get('h', 'N/A')
    Delta = static_params.get('Delta', 'N/A')
    pi = static_params.get('pi', 'N/A')
    gamma = static_params.get('gamma', 'N/A')

    plt.title(f'Dynamic of p and a under Myopic Regulation\n(h={h}, Δ={Delta})', fontsize=16)
    plt.xlabel(' Generation', fontsize=12)
    plt.ylabel('p or a', fontsize=12)
    plt.ylim(0, 1)
    plt.xlim(left=0.5, right=total_generations + 0.5)
    plt.legend(title="Initial Conditions", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置X轴刻度为整数
    if total_generations < 25:
        plt.xticks(np.arange(1, total_generations + 1, step=1))
    else:
        # 如果代数太多，可以适当减少刻度数量
        plt.xticks(np.arange(0, total_generations + 1, step=max(1, total_generations//10)))

    plt.tight_layout()

    filepath = os.path.join(output_dir, "dynamic_p_evolution_comparison.png")
    plt.savefig(filepath, dpi=300)
    print(f"动态演化对比图已保存到: {filepath}")
    plt.show()
