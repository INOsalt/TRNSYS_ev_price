from EVloadDOC import EVload
from EV_cost import EV_earn
from gridinfo import expand_array, prices_real, C_tran, nodes
#from data_output import
from powerflow import OPF
import os
import numpy as np
import pandas as pd

# np.random.seed(12)
# EV_Q1 = np.random.uniform(0.2205, 2, (24,))
# EV_S1 = np.random.uniform(0.2205, 2, (24,))
# EV_2 = np.random.uniform(0.2205, 2, (24,))
# EV_3 = np.random.uniform(0.2205, 2, (24,))
# EV_4 = np.random.uniform(0.2205, 2, (24,))

EV_Q1 = np.array(prices_real)
EV_S1 = np.array(prices_real)
EV_2 = np.array(prices_real)
EV_3 = np.array(prices_real)
EV_4 = np.array(prices_real)

EV_Q1_expanded = expand_array(EV_Q1)
EV_S1_expanded = expand_array(EV_S1)
EV_2_expanded = expand_array(EV_2)
EV_3_expanded = expand_array(EV_3)
EV_4_expanded = expand_array(EV_4)


def main(ev_p, v2g, file_path, nodedata_dict, re_capacity_dict):

    # node_EV_load, mic_EV_load, node_P_total, node_P_basic_and_EV, mic_EV_load_quick, node_slack_load, P_basic_dict\
    #     = EVload(EV_Q1, EV_S1, EV_2, EV_3, EV_4, ev_p, v2g, file_path, nodedata_dict, re_capacity_dict)
    # total_earn = EV_earn(mic_EV_load, mic_EV_load_quick, P_basic_dict,
    #                      EV_Q1_expanded, EV_S1_expanded, EV_2_expanded, EV_3_expanded, EV_4_expanded)

    node_EV_load = pd.read_csv('node_EV_load_nov2g.csv', dtype={'my_column': float})
    node_EV_load.columns = node_EV_load.columns.astype(int)
    print(node_EV_load.columns)
    print(re_capacity_dict.get(0, []))

    opf = OPF(node_EV_load, nodedata_dict, re_capacity_dict)

    generator_costs, system_losses, import_powers, gen_powers = opf.run_ts_opf(ev_p, v2g, file_path)
    cost_import = np.sum(np.array(import_powers) * C_tran)

    # 将 generator_costs 转换为 DataFrame
    generator_all_data = []
    cost_gen = 0  # 初始化成本总和
    for period, costs in enumerate(generator_costs):
        for gen_index, gen_cost in costs:
            generator_all_data.append([period, gen_index, gen_cost])
            cost_gen += gen_cost  # 累加成本
    df_generator_costs = pd.DataFrame(generator_all_data, columns=['Period', 'Generator_Index', 'Cost'])

    #cost = cost_import + cost_gen * 7.2 - total_earn # 买电成本+发电成本-收费 #美元汇率7.2
    lose = np.sum(np.array(system_losses))

    # 定义文件保存路径
    output_dir = os.path.join(file_path, str(ev_p), str(v2g))
    os.makedirs(output_dir, exist_ok=True)  # 确保目标文件夹存在

    # 定义一个辅助函数用于将数据保存到CSV文件
    def save_to_csv(data, filename):
        filepath = os.path.join(output_dir, f'{filename}.csv')

        # 处理字典类型的数据，假设字典的值是向量
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            # 将字典转换为适合的长格式DataFrame
            df = pd.DataFrame([{k: v for k, v in zip(data.keys(), col)} for col in zip(*data.values())])
        # 处理矩阵或数组类型的数据
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        # 处理标量或列表类型的数据
        else:
            df = pd.DataFrame([data])

        # 保存DataFrame到CSV
        df.to_csv(filepath, index=False)


    # # 将各个变量的值保存到CSV文件
    # save_to_csv(node_EV_load, 'node_EV_load')
    # save_to_csv(node_P_total, 'node_P_total')
    # save_to_csv(node_P_basic_and_EV, 'node_P_basic_and_EV')
    # save_to_csv(df_generator_costs, 'generator_costs')
    # save_to_csv(system_losses, 'system_losses')
    # save_to_csv(import_powers, 'import_powers')
    # save_to_csv(total_earn, 'total_earn')
    # save_to_csv(cost, 'cost')
    # save_to_csv(lose, 'lose')

    # return cost, lose, total_earn

ev_p = 800 * 0.15
v2g = 0
file_path = 'test'

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

start_row = 0
end_row = 48

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

main(ev_p, v2g, file_path, nodedata_dict, re_capacity_dict)