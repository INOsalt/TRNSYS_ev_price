from EVloadDOC import EVload_node
from EV_cost import EV_earn
from gridinfo import expand_array, prices_real, C_tran
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


def main(ev_p, v2g, file_path, nodedata_dict, re_capacity_dict, node):

    node_EV_load, mic_EV_load, node_P_total, node_P_basic_and_EV, mic_EV_load_quick, node_slack_load, P_basic_dict\
        = EVload_node(EV_Q1, EV_S1, EV_2, EV_3, EV_4, ev_p, v2g, file_path, nodedata_dict, re_capacity_dict, node)
    total_earn = EV_earn(mic_EV_load, mic_EV_load_quick, P_basic_dict,
                         EV_Q1_expanded, EV_S1_expanded, EV_2_expanded, EV_3_expanded, EV_4_expanded)

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


    # 将各个变量的值保存到CSV文件
    save_to_csv(node_EV_load, 'node_EV_load')
    save_to_csv(node_P_total, 'node_P_total')
    save_to_csv(node_P_basic_and_EV, 'node_P_basic_and_EV')
    save_to_csv(df_generator_costs, 'generator_costs')
    save_to_csv(system_losses, 'system_losses')
    save_to_csv(import_powers, 'import_powers')
    save_to_csv(total_earn, 'total_earn')
    save_to_csv(cost, 'cost')
    save_to_csv(lose, 'lose')

    return cost, lose, total_earn, total_import, total_gen

#main()