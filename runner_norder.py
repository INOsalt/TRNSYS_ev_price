import os
import pandas as pd
from tqdm import tqdm
import gridinfo
from main_noorder import main
import importlib

# 定义EV_penetration和v2g_ratio的可能值
EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]]
v2g = 0

# 创建空的DataFrame，用于存储结果
cost_df = pd.DataFrame(columns=EV_penetration_values)
lose_df = pd.DataFrame(columns=EV_penetration_values)
earn_df = pd.DataFrame(columns=EV_penetration_values)

# 创建存储CSV的目录
data1_dir = 'data_noorder'
os.makedirs(data1_dir, exist_ok=True)

# 循环更新变量并调用main函数，填充DataFrame，并显示进度
total_iterations = len(EV_penetration_values)
with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
    for ev_p in EV_penetration_values:
        cost, lose = main(ev_p, v2g, data1_dir)  # 实际的main函数调用
        cost_df.at[ev_p] = cost
        lose_df.at[ev_p] = lose

        # 更新进度条
        pbar.update(1)

# 保存CSV文件
cost_df.to_csv(os.path.join(data1_dir, 'cost.csv'))
lose_df.to_csv(os.path.join(data1_dir, 'lose.csv'))


