# main.py (最终版，支持独立运行模拟和绘图)

import numpy as np
import argparse  # 导入命令行参数处理库
import sys  # 用于退出程序
import datetime
import os

# 导入我们自己的模块
from module_simulation import perform_comparative_statics
from module_plot import create_plots

# --- 1. 将所有分析配置集中管理 ---
# 我们把每个分析的配置（参数名和取值范围）放到一个字典里，方便调用
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

# --- 2. 定义全局常量 ---
RESULTS_DIR = "results" # 定义结果文件夹的名称

if __name__ == '__main__':
    # --- 3. 在程序开始时，确保结果文件夹存在 ---
    os.makedirs(RESULTS_DIR, exist_ok=True) # 如果文件夹不存在，则创建它；如果已存在，则什么都不做。
    # --- 2. 设置命令行参数解析器 ---
    parser = argparse.ArgumentParser(
        description="运行近交衰退模拟或根据结果绘图。",
        epilog="使用 'python main.py simulate -h' 或 'python main.py plot -h' 查看具体帮助信息。"
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='选择要执行的命令')

    # 创建 'simulate' 命令的解析器
    parser_simulate = subparsers.add_parser('simulate', help='运行一次新的模拟并保存结果。')
    parser_simulate.add_argument(
        'param',
        choices=ANALYSIS_CONFIGS.keys(),
        help=f"选择要进行比较静态分析的参数。可用选项: {list(ANALYSIS_CONFIGS.keys())}"
    )

    # 创建 'plot' 命令的解析器
    parser_plot = subparsers.add_parser('plot', help='从文件加载模拟结果并进行绘图。')
    parser_plot.add_argument(
        'filepath',
        help="包含模拟结果的 .npy 文件路径 (例如: results_h.npy)。"
    )

    args = parser.parse_args()

    # --- 3. 根据命令执行相应操作 ---

    if args.command == 'simulate':
        # --- 执行模拟 ---
        config = ANALYSIS_CONFIGS[args.param]
        analysis_param_name = config['name']
        analysis_param_values = config['values']

        print(f"--- 模式: 模拟 ---")
        print(f"目标参数: {analysis_param_name}")

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

        # 2. 获取当前时间并格式化成字符串
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 使用 os.path.join 来构建跨平台兼容的文件路径
        filename = f"results_{analysis_param_name}_{timestamp}.npy"
        output_filepath = os.path.join(RESULTS_DIR, filename)

        np.save(output_filepath, analysis_results)
        print(f"\n模拟完成！结果已保存到文件: {output_filepath}")

    elif args.command == 'plot':
        # --- 执行绘图 ---
        print(f"--- 模式: 绘图 ---")
        print(f"加载结果文件: {args.filepath}")

        try:
            loaded_results = np.load(args.filepath, allow_pickle=True).item()
            # 将结果文件夹的路径也传递给绘图函数
            create_plots(loaded_results, RESULTS_DIR)
        except FileNotFoundError:
            print(f"错误: 文件 '{args.filepath}' 未找到。请检查路径和文件名。")
            sys.exit(1)





#
# # =============================================================================
# #  MAIN EXECUTION BLOCK
# # =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
#
# from module_simulation import perform_comparative_statics
# from module_plot import create_plots
#
# if __name__ == '__main__':
#     # --- 1. 定义所有参数的“基础值”或“默认值” ---
#     BASE_PARAMS = {
#         "population_size": 10000,
#         "num_generations": 10,
#         "initial_A_proportion": 0.5,
#         "h": 0.8,  # 显性系数
#         "Delta": 0.3,  # 选择优势
#         "pi": 0.3,  # 疾病发生率
#         "gamma": 0.8  # 疾病严重性
#     }
#
#     # 定义 'a' 的优化范围 (这个通常在所有分析中保持不变)
#     ai_values_to_scan = np.linspace(0.5, 1, 51)
#
#     # --- 2. 选择并配置你本次想做的分析 ---
#     # --- 只需取消注释你想要运行的那一组即可 ---
#
#     # 【分析案例1：对 h 进行比较静态分析】
#     # analysis_param_name = 'h'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # 【分析案例2：对 Delta (选择优势) 进行比较静态分析】
#     # analysis_param_name = 'Delta'
#     # analysis_param_values = np.linspace(0.1, 0.5, 41)
#
#     # 【分析案例3：对 pi (疾病发生率) 进行比较静态分析】
#     # analysis_param_name = 'pi'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # 【分析案例4：对 gamma (疾病严重性) 进行比较静态分析】 (新增)
#     # analysis_param_name = 'gamma'
#     # analysis_param_values = np.linspace(0, 1, 101)
#
#     # 【分析案例5：对 num_generations (模拟代数) 进行比较静态分析】 (新增)
#     # analysis_param_name = 'num_generations'
#     # # 对于整数参数，使用 np.arange 更直观
#     # analysis_param_values = np.arange(2, 50, 1, dtype=int)
#
#     # 【分析案例6：对 initial_A_proportion (初始基因频率) 进行比较静态分析】 (新增)
#     analysis_param_name = 'initial_A_proportion'
#     analysis_param_values = np.linspace(0, 1, 101)
#
#     # --- 3. 准备参数并执行 ---
#
#     # 从基础参数中复制一份，作为本次分析中保持不变的参数
#     static_params_for_run = BASE_PARAMS.copy()
#     # 从“固定”参数字典中移除即将要扫描的那个参数的条目
#     if analysis_param_name in static_params_for_run:
#         del static_params_for_run[analysis_param_name]
#
#     # 执行通用的分析函数
#     analysis_results = perform_comparative_statics(
#         static_params=static_params_for_run,
#         sweep_param_name=analysis_param_name,
#         sweep_param_values=analysis_param_values,
#         ai_values=ai_values_to_scan
#     )
#
#     # --- 4. 绘图 ---
#     create_plots(analysis_results)