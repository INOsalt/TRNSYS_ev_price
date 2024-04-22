import numpy as np
import pandas as pd
import os

#=================================
num_nodes = 40
charge_ratio = 0.06
# EV_penetration = 800 * 0.15
# v2g_ratio = 0

prices_real = [
    0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
    1.23, 1.23,
    1.51,
    1.67,
    1.23, 1.23,
    1.51,
    1.67, 1.67,
    1.51, 1.51,
    1.23, 1.23, 1.23, 1.23, 1.23
]

# 定义节点列表
nodes = [101, 102, 103, 104, 105, 106, 201, 202, 203, 204, 205, 206, 207, 208, 209, 301, 302, 303, 304, 305, 306,
         307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 401, 402, 403, 404, 405, 406, 407]

# 创建一个从节点编号到索引的映射
node_mapping = {node: index for index, node in enumerate(nodes)}
# 反向映射：从矩阵索引到节点编号
reverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

# 初始化所有车辆都停泊在起点，没有车辆在移动状态
df_start = pd.read_csv('start.csv')
df_end = pd.read_csv('end.csv')
# Extract columns to lists
start_points = df_start['bus_i'].to_list()
# print(start_points)
EV_num = df_start['EV_num'].to_list()
end_points = df_end['bus_i'].to_list()

# 初始车辆分布
initial_EV = np.zeros(2 * num_nodes)
starts_indices = np.array([node_mapping[point] for point in start_points])
for i, index in enumerate(starts_indices):
    initial_EV[index] += EV_num[i]

#直接相连的边: {(12, 2), (5, 4), (3, 5), (3, 2), (15, 3), (6, 1), (1, 4), (2, 1), (26, 5), (5, 0), (2, 5), (25, 3), (10, 3), (2, 4), (4, 0)}
#间隔一个点连接的边: {(2, 1), (2, 5), (10, 3), (26, 25), (12, 2), (7, 6), (12, 6), (10, 12), (25, 26), (5, 0), (16, 15), (16, 25), (12, 10), (24, 10), (3, 2), (5, 4), (15, 3), (24, 15), (1, 4), (9, 10), (6, 12), (25, 3), (3, 5), (8, 12), (6, 1), (26, 5), (28, 25), (2, 4)}

# 读取转移矩阵
# 设置包含CSV文件的文件夹路径
folder_path = 'TMhalfhour'

# 初始化一个空字典来存储矩阵
transition_matrices = {}
keys = np.arange(0, 48)

# 遍历文件夹中的所有文件
i = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        matrix_df = pd.read_csv(file_path, header=None)
        # 将DataFrame转换为NumPy数组
        matrix = matrix_df.to_numpy()
        #print(f"Matrix {filename} size: {matrix.shape}")#检查矩阵形状
        matrix_name = keys[i]
        i = i + 1
        # 将矩阵存储到字典中
        transition_matrices[matrix_name] = matrix

#主网电价
C_buy = [0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205,
         0.5802, 0.5802, 0.9863, 0.9863, 0.5802, 0.5802, 0.9863, 0.9863,
         0.9863, 0.9863, 0.9863, 0.5802, 0.5802, 0.5802, 0.5802, 0.5802]
C_sell = [0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
          0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
          0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453]
C_buy_business = [
    0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867,
    0.7093, 0.7093,
    1.1864, 1.1864,
    0.7093, 0.7093,
    1.1864, 1.1864, 1.1864, 1.1864, 1.1864,
    0.7093, 0.7093, 0.7093, 0.7093, 0.7093
]
C_tran = 0.1834

# 定义一个函数来将每个元素重复两次并扩展数组到48长度
def expand_array(arr):
    return np.repeat(arr, 2)
#=========潮流计算数据处理 牛拉法需要的格式=========================
## 读取branch数据的CSV文件
branch_df = pd.read_csv('grid/branch_data.csv')
# 将branch_df转换为字典，其中每个值都是NumPy数组
branch = {col: branch_df[col].values for col in branch_df.columns}

bus_df = pd.read_csv('grid/bus_data.csv')
# 将bus_df转换为字典，其中每个值都是NumPy数组
bus = {col: bus_df[col].values for col in bus_df.columns}

gen_df = pd.read_csv('grid/gen_data_large.csv')
# 将gen_df转换为字典，其中每个值都是NumPy数组
gen = {col: gen_df[col].values for col in gen_df.columns}
# print(gen)
# gen_df1 = pd.read_csv('grid/gen_data_large.csv')
# # 将gen_df转换为字典，其中每个值都是NumPy数组
# gen1 = {col: gen_df1[col].values for col in gen_df1.columns}
# print(gen1)

# pvwt_reactive_df = pd.read_csv('grid/pvwt_reactive.csv')
# pvwt_reactive = {col: pvwt_reactive_df[col].values for col in pvwt_reactive_df.columns}
#
# # 节点负荷矩阵# %Nodedata=[Bus ID	kW  KVAR ]
# # 读取基础负荷数据
# base_load_df = pd.read_csv('grid/base_load.csv')
# base_load_data = base_load_df.to_numpy()
# # 读取负荷百分比数据
# load_percent_df = pd.read_csv('grid/load_percent.csv')
# load_percent_data = load_percent_df[['hour', 'Wkdy']].to_numpy()

# #=========处理微电网相关数据======================
# # 初始化两个字典，用于存储每个微电网的PV和WT发电量向量
# microgrid_pv = {0: np.zeros(24), 1: np.zeros(24), 2: np.zeros(24), 3: np.zeros(24)}
# microgrid_wt = {0: np.zeros(24), 1: np.zeros(24), 2: np.zeros(24), 3: np.zeros(24)}
#
# 定义一个函数，用于确定Bus ID属于哪个微电网
def microgrid_id(bus_id):
    if 100 <= bus_id < 200:
        return 0
    elif 200 <= bus_id < 300:
        return 1
    elif 300 <= bus_id < 400:
        return 2
    elif 400 <= bus_id < 500:
        return 3
#
# # 遍历每个小时，更新微电网的PV和WT发电量
# for hour in range(24):
#     # 处理PV发电能力
#     for bus_id, pg in pv_capacity_dict[hour]:
#         mg_id = microgrid_id(bus_id)
#         microgrid_pv[mg_id][hour] += pg
#
#     # 处理WT发电能力
#     for bus_id, pg in wt_capacity_dict[hour]:
#         mg_id = microgrid_id(bus_id)
#         microgrid_wt[mg_id][hour] += pg
#
# # 初始化一个字典来存储每个微电网每小时的总负荷
# microgrid_load_dict = {0: np.zeros(24), 1: np.zeros(24), 2: np.zeros(24), 3: np.zeros(24)}
#
# # 遍历nodedata_dict中的每小时节点负荷矩阵
# for hour, loads in nodedata_dict.items():
#     # 遍历该小时每个节点的负荷数据
#     for load in loads:
#         bus_id, kW, _ = load  # 假设负荷矩阵中的列分别是Bus ID、kW、KVAR
#         mg_id = microgrid_id(int(bus_id))  # 使用不同的变量名来存储函数返回值
#         if mg_id:  # 检查mg_id是否有效（不是None）
#             microgrid_load_dict[mg_id][hour] += kW  # 累加对应微电网的kW负荷 # 累加对应微电网的kW负荷

# #=========映射处理=========
# branch['fbus'] = np.array([node_mapping[node] for node in branch['fbus']])
# branch['tbus'] = np.array([node_mapping[node] for node in branch['tbus']])
# bus['bus_i'] = np.array([node_mapping[node] for node in bus['bus_i']])
# gen['gen_bus'] = np.array([node_mapping[node] for node in gen['gen_bus']])
# 更新 pv_capacity_dict 中的节点编号
# mapped_pv_dict = {}
# for hour, capacities in pv_capacity_dict.items():
#     # 创建一个新的数组用于存储映射后的数据
#     mapped_capacities = np.empty_like(capacities)
#     for i, (bus_id, pg) in enumerate(capacities):
#         # 映射节点编号到索引
#         mapped_bus_id = node_mapping[bus_id]
#         mapped_capacities[i] = [mapped_bus_id, pg]
#     mapped_pv_dict[hour] = mapped_capacities
#
# # 更新 wt_capacity_dict 中的节点编号
# mapped_wt_dict = {}
# for hour, capacities in wt_capacity_dict.items():
#     # 创建一个新的数组用于存储映射后的数据
#     mapped_capacities = np.empty_like(capacities)
#     for i, (bus_id, pg) in enumerate(capacities):
#         # 映射节点编号到索引
#         mapped_bus_id = node_mapping[bus_id]
#         mapped_capacities[i] = [mapped_bus_id, pg]
#     mapped_wt_dict[hour] = mapped_capacities
#
# # 更新节点负荷数据中的节点编号
# mapped_nodedata_dict = {}
#
# for hour, loads in nodedata_dict.items():
#     # 创建一个新数组用于存储映射后的数据
#     # 假设loads的结构是[Bus ID, kW, KVAR]
#     mapped_loads = np.empty_like(loads)
#     for i, (bus_id, kW, KVAR) in enumerate(loads):
#         # 使用node_mapping映射节点编号到索引
#         mapped_bus_id = node_mapping.get(bus_id, None)  # 获取映射后的节点索引，如果不存在则返回None
#         if mapped_bus_id is not None:
#             # 更新节点编号并保持其他数据不变
#             mapped_loads[i] = [mapped_bus_id, kW, KVAR]
#         else:
#             # 如果在node_mapping找不到对应的节点编号，则保持原始编号
#             # 或者根据需要处理这种情况，例如通过打印警告
#             print(f"Warning: Bus ID {bus_id} not found in node_mapping.")
#     # 将映射后的数据存储到新字典中
#     mapped_nodedata_dict[hour] = mapped_loads


