import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# --- 解决中文显示问题的代码 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False


class GeneticOptimizer:
    def __init__(self, h=0.5, Delta=1.0, beta=0.95, c=1.0, cost_type='quadratic'):
        """
        初始化遗传优化问题

        参数:
        h: 显性占优系数
        Delta: 产量提升系数
        beta: 贴现因子
        c: 成本函数系数
        cost_type: 成本函数类型 ('quadratic' 或 'exponential')
        """
        self.h = h
        self.Delta = Delta
        self.beta = beta
        self.c = c
        self.cost_type = cost_type

        # 状态空间离散化
        self.n_points = 501  # 网格点数量
        self.p_grid = np.linspace(0, 1, self.n_points)
        self.dp = self.p_grid[1] - self.p_grid[0]

        # 初始化值函数
        self.V = np.zeros(self.n_points)
        self.V_new = np.zeros(self.n_points)
        self.policy = np.zeros(self.n_points)

    def cost_function(self, a, p):
        """成本函数 C(x_t)"""
        x = a * p + (1 - a) * (1 - p)  # 纯合子比例
        if self.cost_type == 'quadratic':
            return self.c * ((a * p) ** 2) * 2 + self.c * (((1 - a) * (1 - p)) ** 2)* 2
            # return self.c * x ** 4 * 4
        elif self.cost_type == 'exponential':
            return self.c * (np.exp(x) - 1)
        else:
            raise ValueError("cost_type must be 'quadratic' or 'exponential'")

    def cost_derivative(self, a, p):
        """成本函数的导数 C'(x_t)"""
        x = a * p + (1 - a) * (1 - p)
        if self.cost_type == 'quadratic':
            return 2 * self.c * x
        elif self.cost_type == 'exponential':
            return self.c * np.exp(x)

    def immediate_payoff(self, a, p):
        """即期收益函数"""
        heterozygous = a * (1 - p) + (1 - a) * p  # 杂合子比例
        return 1 + (a * p + self.h * heterozygous) * self.Delta - self.cost_function(a, p)

    def state_transition(self, a, p):
        """状态转移方程"""
        return 0.5 * (a + p)

    def interpolate_value(self, p):
        """对值函数进行插值"""
        if p <= 0:
            return self.V[0]
        elif p >= 1:
            return self.V[-1]
        else:
            # 线性插值
            idx = int(p / self.dp)
            if idx >= self.n_points - 1:
                return self.V[-1]
            weight = (p - self.p_grid[idx]) / self.dp
            return self.V[idx] * (1 - weight) + self.V[idx + 1] * weight

    def bellman_operator(self, p_idx):
        """贝尔曼算子：对给定状态p找到最优控制a"""
        p = self.p_grid[p_idx]

        def objective(a):
            """目标函数：-(即期收益 + 贴现未来价值)"""
            if a < 0 or a > 1:
                return 1e10  # 惩罚违反约束的解

            payoff = self.immediate_payoff(a, p)
            next_p = self.state_transition(a, p)
            future_value = self.interpolate_value(next_p)

            return -(payoff + self.beta * future_value)

        # 使用一阶条件作为初始猜测
        # try:
        #     # 从一阶条件求解最优a
        #     result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        #     optimal_a = result.x
        #     optimal_value = -result.fun
        # except:
            # 如果优化失败，使用网格搜索
        a_candidates = np.linspace(0, 1, 501)
        values = [-objective(a) for a in a_candidates]
        best_idx = np.argmax(values)
        optimal_a = a_candidates[best_idx]
        optimal_value = values[best_idx]

        return optimal_a, optimal_value

    def solve_foc_directly(self, p):
        """直接求解一阶条件"""

        def foc_equation(a):
            """一阶条件方程"""
            if a < 0 or a > 1:
                return 1e10

            # 计算当前期的一阶条件
            term1 = (p + self.h * (1 - 2 * p)) * self.Delta
            term2 = self.cost_derivative(a, p) * (2 * p - 1)

            # 计算未来价值的边际效应（数值微分）
            eps = 1e-6
            next_p = self.state_transition(a, p)
            next_p_plus = self.state_transition(a + eps, p)

            dV_dp = (self.interpolate_value(next_p_plus) - self.interpolate_value(next_p)) / (eps * 0.5)

            return term1 - term2 + self.beta * 0.5 * dV_dp

        try:
            # 寻找使一阶条件为零的a值
            result = fsolve(foc_equation, 0.5)
            a_optimal = np.clip(result[0], 0, 1)
            return a_optimal
        except:
            return 0.5  # 如果求解失败，返回中间值

    def value_function_iteration(self, max_iter=5000, tol=1e-6, verbose=True):
        """值函数迭代算法"""
        for iteration in range(max_iter):
            # 对每个状态点进行贝尔曼更新
            for i in range(self.n_points):
                optimal_a, optimal_value = self.bellman_operator(i)
                self.V_new[i] = optimal_value
                self.policy[i] = optimal_a

            # 检查收敛性
            max_diff = np.max(np.abs(self.V_new - self.V))

            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Max difference = {max_diff:.6f}")

            if max_diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            # 更新值函数
            self.V = self.V_new.copy()

        return iteration + 1

    def simulate_trajectory(self, p0, T=20):
        """模拟最优轨迹"""
        p_trajectory = np.zeros(T + 1)
        a_trajectory = np.zeros(T)

        p_trajectory[0] = p0

        for t in range(T):
            # 从最优策略中插值得到a_t
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
                    a_optimal = self.policy[idx] * (1 - weight) + self.policy[idx + 1] * weight

            a_trajectory[t] = a_optimal
            p_trajectory[t + 1] = self.state_transition(a_optimal, p_current)

        return p_trajectory, a_trajectory

    def plot_results(self, p0_list=[0.2, 0.5, 0.8]):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # --- 新增代码：为整个图像添加主标题 ---
        # fig.suptitle(f'Parameters: h={self.h}, Delta ={self.Delta}, beta = {self.beta},'
        #              f' c={self.c}, cost_type = {self.cost_type}, power_coef = 2222',
        #              fontsize=16, fontweight='bold')
        fig.suptitle(f'Parameters: h={self.h}, Delta ={self.Delta}, beta = {self.beta},'
                     f' cost_type = {self.cost_type}', fontsize=16, fontweight='bold')


        # 1. 值函数
        axes[0, 0].plot(self.p_grid, self.V, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('p (proportion of allele A)')
        axes[0, 0].set_ylabel('V(p) ')
        axes[0, 0].set_title('Value Function')
        axes[0, 0].grid(True)

        # 2. 最优策略
        axes[0, 1].plot(self.p_grid, self.policy, 'r-', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x') #绘制对角线
        axes[0, 1].set_xlim(0,1)
        axes[0, 1].set_ylim(0,1)
        axes[0, 1].set_xlabel('p(proportion of allele A)')
        axes[0, 1].set_ylabel('a (Optimal Policy)')
        axes[0, 1].set_title('Optimal Policy Function')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 轨迹模拟 - p的演化
        colors = ['blue', 'green', 'red']
        for i, p0 in enumerate(p0_list):
            p_traj, a_traj = self.simulate_trajectory(p0, T=10)
            axes[1, 0].plot(p_traj, color=colors[i], linewidth=2,
                            label=f'Initial p={p0}', marker='o', markersize=3)

        axes[1, 0].set_ylim(0,1)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('p_t (proportion of allele A)')
        axes[1, 0].set_title('Dynamic of p')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 轨迹模拟 - a的演化
        for i, p0 in enumerate(p0_list):
            p_traj, a_traj = self.simulate_trajectory(p0, T=10)
            axes[1, 1].plot(a_traj, color=colors[i], linewidth=2,
                            label=f'Initial p={p0}', marker='s', markersize=3)
        axes[1, 1].set_ylim(0,1)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('a_t (Proportion of allele A in AI)')
        axes[1, 1].set_title('Dynamic of a')
        axes[1, 1].legend()
        axes[1, 1].grid(True)


        plt.tight_layout()
        plt.show()

    def plot_bellman_objective(self, p_values_to_plot=[0.2, 0.5, 0.8]):
        """
        可视化贝尔曼方程右侧(RHS)的目标函数。

        对于给定的状态p, 绘制 payoff(a,p) + beta * V(p_next) 关于 a 的函数图像。
        这个图像的最高点对应的a就是该状态p下的最优策略a*(p)。
        """
        # 确保值函数已经收敛，否则此图无意义
        if np.all(self.V == 0):
            print("警告: 值函数尚未计算，请先运行 value_function_iteration。")
            # 可以在此返回或继续，但V(p)将全为0

        a_grid = np.linspace(0, 1, 201)  # 为a创建一个更精细的网格以便绘图

        plt.figure(figsize=(10, 7))

        # 为了更平滑地插值V(p_next)
        # 我们使用三次样条插值，这对于可视化凹凸性至关重要
        # 注意：这里我们假设V已经收敛
        V_interpolator = interp1d(self.p_grid, self.V, kind='cubic', fill_value="extrapolate")

        for p_val in p_values_to_plot:
            objective_values = []
            for a in a_grid:
                payoff = self.immediate_payoff(a, p_val)
                next_p = self.state_transition(a, p_val)
                future_value = V_interpolator(next_p)  # 使用平滑插值

                # 这就是贝尔曼方程的右侧 (RHS)
                total_value = payoff + self.beta * future_value
                objective_values.append(total_value)

            # 绘制曲线
            plt.plot(a_grid, objective_values, label=f'p = {p_val:.2f}')

            # 找到并标记最优点
            max_val_idx = np.argmax(objective_values)
            optimal_a = a_grid[max_val_idx]
            max_val = objective_values[max_val_idx]
            plt.plot(optimal_a, max_val, 'o', markersize=8,
                     label=f'a*({p_val:.2f}) ≈ {optimal_a:.2f}')

        plt.xlabel('a (策略变量: 人工繁殖A的比例)', fontsize=12)
        plt.ylabel(r'即期与未来收益之和: $R(a, p) + \beta V(p\')$', fontsize=12)
        plt.title('贝尔曼方程目标函数可视化', fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.show()

    def analyze_steady_state(self):
        """分析稳态"""
        print("稳态分析:")
        print("-" * 40)

        # 寻找稳态点（p = a的点）
        steady_states = []
        for i in range(self.n_points):
            p = self.p_grid[i]
            a = self.policy[i]
            if abs(p - a) < 0.01:  # 接近稳态的条件
                steady_states.append((p, a))

        if steady_states:
            print("发现的稳态点:")
            for p, a in steady_states:
                print(f"  p* = {p:.3f}, a* = {a:.3f}")
        else:
            print("未发现明显的稳态点")

        # 分析政策函数的性质
        print(f"\n政策函数分析:")
        print(f"  最小控制值: {np.min(self.policy):.3f}")
        print(f"  最大控制值: {np.max(self.policy):.3f}")
        print(f"  平均控制值: {np.mean(self.policy):.3f}")


# 使用示例
if __name__ == "__main__":
    # 设置参数
    optimizer = GeneticOptimizer(
        h=0.6,  # 显性占优系数
        Delta=0.4,  # 产量提升系数
        beta=0.95,  # 贴现因子
        c=1.0,  # 成本函数系数
        cost_type='quadratic'  # 成本函数类型
    )

    print("开始求解动态优化问题...")
    print(f"参数设置: h={optimizer.h}, Δ={optimizer.Delta}, β={optimizer.beta}, c={optimizer.c}")

    # 求解
    iterations = optimizer.value_function_iteration(max_iter=5000, tol=1e-6)

    # 分析结果
    optimizer.analyze_steady_state()

    # 绘制结果
    optimizer.plot_results()

    print("\n正在生成贝尔曼方程目标函数的可视化图像...")
    optimizer.plot_bellman_objective(p_values_to_plot=[0.2, 0.5, 0.8])

    # 显示一些关键点的策略
    print("\n关键点的最优策略:")
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for p in test_points:
        idx = int(p / optimizer.dp)
        if idx >= optimizer.n_points:
            idx = optimizer.n_points - 1
        print(f"  p = {p:.2f}: a* = {optimizer.policy[idx]:.3f}, V = {optimizer.V[idx]:.3f}")