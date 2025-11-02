from module_simulation import calculate_yield_one_generation
import numpy as np
import matplotlib.pyplot as plt


population_size = 100000
initial_a_proportion = 0.65
h = 0.8
Delta = 0.5
pi = 1.0
gamma = 0.8

ai_values = np.linspace(0, 1, 101)

p_A_maternal = initial_a_proportion
p_a_maternal = 1 - p_A_maternal

yields= []

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
    yields.append(yield_val)

print(yields)


# --- 绘图代码 ---
# 创建图形
plt.plot(ai_values, yields, label='Expected Yield')

# 添加图表标题和坐标轴标签
plt.title('Relationship between a and Yield when p = 0.65')
plt.xlabel('Policy variable a: the Proportion of \'G\' Allele used in AI')
plt.ylabel('Expected Yield')

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

# --- 修改此处 ---
# 将保存文件改为直接显示图形
plt.show()