from powerflow import powerflow
from mainDOC import MGO
from EVloadDOC import EVload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gridinfo import EV_penetration, expand_array
import os


# 基本定义
N = 3 # 初始种群个数
d = 144 # 空间维数
ger = 50 # 最大迭代次数

# 位置限制和速度限制
# 固定随机数生成器的种子
np.random.seed(42)
# 生成位置限制的下限和上限
plimit_lower = np.random.rand(d)  # 随机生成下限
plimit_upper = plimit_lower + np.random.rand(d) * 0.5  # 在下限基础上加上一个正的偏移量，确保上限大于下限

# 将下限和上限合并为plimit数组
plimit = np.vstack((plimit_lower, plimit_upper))

vlimit = np.array([1 + np.zeros(d), 1 + np.zeros(d)])  # 速度限制设为 -1 到 1

w = 0.8  # 惯性权重
c1 = 0.5  # 自我学习因子
c2 = 0.5  # 群体学习因子

# 初始化种群的位置和速度
x = np.zeros((N, d))
for i in range(d):
    x[:, i] = plimit[0, i] + (plimit[1, i] - plimit[0, i]) * np.random.rand(N)

v = np.random.rand(N, d)  # 初始化速度

xm = np.copy(x)  # 每个个体的历史最佳位置
ym = np.zeros(d)  # 种群的历史最佳位置
fxm = np.zeros(N) + 125000  # 每个个体的历史最佳适应度
fym = float('inf')  # 种群历史最佳适应度

def obj_all(x, iter):
    """
    简化的目标函数，用于测试PSO算法。
    :param x: 优化变量数组
    :param iter: 当前迭代次数（此例中未使用）
    :return: 两个目标函数的值
    """
    # 假设x是一个一维数组，我们只取第一个元素进行计算
    F1 = x[0] ** 2
    F2 = (x[0] - 2) ** 2
    return F1, F2


def dominates(a, b):
    """
    判断解a是否支配解b。

    :param a: 解a的目标函数值，格式为[F1值, F2值]。
    :param b: 解b的目标函数值，格式为[F1值, F2值]。
    :return: 如果a支配b，则返回True；否则返回False。
    """
    better_in_one = False
    for i in range(len(a)):
        if a[i] > b[i]:  # 如果a在任一目标上比b差，则a不支配b
            return False
        elif a[i] < b[i]:  # 如果a在任一目标上比b好，则记录a至少在一个目标上比b好
            better_in_one = True
    return better_in_one  # 如果a至少在一个目标上比b好且没有在任何目标上比b差，则a支配b


# 初始化每个个体的当前适应度
fx = np.zeros(N)

# 迭代更新开始
iter = 1
record = np.zeros(ger)

# 假设已经有了初始化粒子群的代码

# 初始化帕累托前沿列表
pareto_front = []

# 迭代更新开始
for iter in range(ger):
    print("Iteration:", iter)
    # 用于本次迭代中更新帕累托前沿的临时列表
    new_pareto_front = []

    for n in range(N):
        # 计算每个粒子的两个目标函数值
        F1, F2 = obj_all(x[n, :], iter)
        solution = (x[n, :], [F1, F2])  # 将解的表示和目标函数值打包

        is_dominated = False
        remove_indices = []

        for idx, pf in enumerate(pareto_front):
            if dominates(solution[1], pf[1]):  # 使用目标函数值比较
                remove_indices.append(idx)
            elif dominates(pf[1], solution[1]):
                is_dominated = True
                break

        for idx in sorted(remove_indices, reverse=True):
            del pareto_front[idx]

        if not is_dominated:
            new_pareto_front.append(solution)  # 添加包含目标函数值的解

    # 更新帕累托前沿
    pareto_front.extend(new_pareto_front)

    # 速度更新
    v = w * v + c1 * np.random.rand() * (xm - x) + c2 * np.random.rand() * (np.tile(ym, (N, 1)) - x)

    # 边界速度处理
    v = np.clip(v, -vlimit[0, :], vlimit[0, :])

    # 位置更新
    x = x + v

    # 边界位置处理
    for i in range(d):
        x[:, i] = np.clip(x[:, i], plimit[0, i], plimit[1, i])

    record[iter - 1] = fym
    iter += 1

# 结果展示


# 将帕累托解集输出到 CSV 文件
df = pd.DataFrame(pareto_front, columns=['Objective 1', 'Objective 2'])
# 设置文件路径
csv_file_path = 'data/pareto_front.csv'  # 注意去掉了开头的 '/', 使路径相对于当前工作目录

# 确保目录存在
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# 保存到CSV
df.to_csv(csv_file_path, index=False)

print(f"File saved to {csv_file_path}")

# 首先提取 F1 和 F2 的值
objective_1_values = [item[1][0] for item in pareto_front]  # F1 值
objective_2_values = [item[1][1] for item in pareto_front]  # F2 值

# 绘制帕累托前沿
plt.figure(figsize=(8, 6))
plt.scatter(objective_1_values, objective_2_values, color='blue', label='Pareto Front')
plt.title('Pareto Front')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.legend()
plt.grid(True)
plt.show()



