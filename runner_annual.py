import os
import pandas as pd
from tqdm import tqdm
from gridinfo import nodes
from main_module import main
import importlib
import pandas as pd
import numpy as np

# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #
ev_p = 800 * 0.15
v2g = 0.5
data1_dir = 'data_annual'

# 读取CSV文件
folder_path = 'annual/'
p_from_grid_filename = folder_path + 'P_from_grid_kW_total.csv'
reactive_power_filename = folder_path + 'reactive_power_total.csv'
p_to_grid_filename = folder_path + 'P_to_grid_kW_total.csv'

p_from_grid_df = pd.read_csv(p_from_grid_filename, dtype={'my_column': float})
reactive_power_df = pd.read_csv(reactive_power_filename, dtype={'my_column': float})
p_to_grid_df = pd.read_csv(p_to_grid_filename, dtype={'my_column': float})

# nodes = list(p_from_grid_df.columns)  # 假设列名为节点编号

annual_costs = []
annual_loses = []
annual_earns = []
annual_imports = []
annual_gens = []

for day in range(365):
    start_row = day * 48
    end_row = (day + 1) * 48

    # 获取当日数据
    p_from_grid_day = p_from_grid_df.iloc[start_row:end_row]
    reactive_power_day = reactive_power_df.iloc[start_row:end_row]
    p_to_grid_day = p_to_grid_df.iloc[start_row:end_row]

    nodedata_dict = {}
    re_capacity_dict = {}

    for half_hour in range(48):
        # 获取当前半小时段的有功功率和无功功率数据
        active_power_row = p_from_grid_day.iloc[half_hour].values
        reactive_power_row = reactive_power_day.iloc[half_hour].values
        re_power_row = p_to_grid_day.iloc[half_hour].values
        # 对每个半小时构建矩阵
        nodedata_matrix = np.column_stack((nodes, active_power_row, reactive_power_row))
        re_matrix = np.column_stack((nodes, re_power_row))
        nodedata_dict[half_hour] = nodedata_matrix
        re_capacity_dict[half_hour] = re_matrix

    # 计算成本
    cost, lose, total_earn, total_import, total_gen = main(ev_p, v2g, data1_dir, nodedata_dict, re_capacity_dict)
    annual_costs.append(cost)
    annual_loses.append(lose)
    annual_earns.append(total_earn)
    annual_imports.append(total_import)
    annual_gens.append(total_gen)
    print(f'第{day + 1}天的成本: {cost}')
    print(f'第{day + 1}天的网损: {lose}')
    print(f'第{day + 1}天的收入: {total_earn}')
    print(f'第{day + 1}天的进口: {total_import}')
    print(f'第{day + 1}天的发电: {total_gen}')
#
# 将成本数据保存到CSV文件
cost_annual_df = pd.DataFrame({'Day': range(1, 366), 'Cost': annual_costs})
cost_annual_df.to_csv(folder_path + 'cost_annual.csv', index=False)
# 将成本数据保存到CSV文件
lose_annual_df = pd.DataFrame({'Day': range(1, 366), 'Cost': annual_loses})
lose_annual_df.to_csv(folder_path + 'lose_annual.csv', index=False)
# 将成本数据保存到CSV文件
earn_annual_df = pd.DataFrame({'Day': range(1, 366), 'Cost': annual_earns})
earn_annual_df.to_csv(folder_path + 'earn_annual.csv', index=False)
# 将成本数据保存到CSV文件
import_annual_df = pd.DataFrame({'Day': range(1, 366), 'Cost': annual_imports})
import_annual_df.to_csv(folder_path + 'earn_annual.csv', index=False)
# 将成本数据保存到CSV文件
gen_annual_df = pd.DataFrame({'Day': range(1, 366), 'Cost': annual_gens})
gen_annual_df.to_csv(folder_path + 'earn_annual.csv', index=False)
#
#
# # 创建空的DataFrame，用于存储结果
# cost_df = pd.DataFrame(index=v2g_ratio_values, columns=EV_penetration_values)
# lose_df = pd.DataFrame(index=v2g_ratio_values, columns=EV_penetration_values)
# earn_df = pd.DataFrame(index=v2g_ratio_values, columns=EV_penetration_values)
#
# # 创建存储CSV的目录
# data1_dir = 'data2'
# os.makedirs(data1_dir, exist_ok=True)
#
# # 循环更新变量并调用main函数，填充DataFrame，并显示进度
# total_iterations = len(EV_penetration_values) * len(v2g_ratio_values)
# with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
#     for ev_p in EV_penetration_values:
#         for v2g in v2g_ratio_values:
#             cost, lose, total_earn = main(ev_p, v2g, data1_dir)
#             cost_df.at[v2g, ev_p] = cost
#             lose_df.at[v2g, ev_p] = lose
#             earn_df.at[v2g, ev_p] = total_earn
#
#             # 更新进度条
#             pbar.update(1)
#
# # 保存CSV文件
# cost_df.to_csv(os.path.join(data1_dir, 'cost.csv'))
# lose_df.to_csv(os.path.join(data1_dir, 'lose.csv'))
# earn_df.to_csv(os.path.join(data1_dir, 'earn.csv'))


