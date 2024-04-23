import os
from tqdm import tqdm
from gridinfo import nodes, C_tran
from main_module import main
import importlib
import pandas as pd
import numpy as np
import ast
from annual_powerflow import OPF
from Basic_cost import Basic_earn

# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #


data1_dir = 'data_annual'

#场景
sen_type = 'OnlyPV'
strategy_type = 'v2g'
# 读取CSV文件
folder_path = 'annual_' + sen_type + '/'
p_from_grid_filename = folder_path + 'P_from_grid_kW_total.csv'
reactive_power_filename = folder_path + 'reactive_power_total.csv'
p_to_grid_filename = folder_path + 'P_to_grid_kW_total.csv'

p_from_grid_df = pd.read_csv(p_from_grid_filename, dtype={'my_column': float})
reactive_power_df = pd.read_csv(reactive_power_filename, dtype={'my_column': float})
p_to_grid_df = pd.read_csv(p_to_grid_filename, dtype={'my_column': float})


def save_to_csv(data, output_file_path, filename):
    filepath = os.path.join(output_file_path, f'{filename}.csv')

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

def daily_powerflow(ev_p,v2g):
    # 读取CSV文件
    ev_opt_file_path = f'powerflow_{sen_type}_{strategy_type}/daily_total_{sen_type}_{strategy_type}_{ev_p}.csv'
    daily_EV_OPT = pd.read_csv(ev_opt_file_path)
    # 删除第一列
    daily_EV_OPT = daily_EV_OPT.drop(daily_EV_OPT.columns[0], axis=1)  # 删除第一列

    # 将每个元素中的字符串转换为数组
    daily_EV_OPT = daily_EV_OPT.applymap(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # nodes = list(p_from_grid_df.columns)  # 假设列名为节点编号

    annual_costs = []
    annual_loses = []
    annual_earns = []
    annual_imports = []
    annual_gens = []
    annual_curtailed = []
    error_day = []

    # 创建文件夹路径
    output_file_path = f'annual_powerflow_{sen_type}_{strategy_type}'
    if not os.path.exists(output_file_path):
        # 如果不存在，则创建文件夹
        os.makedirs(output_file_path)

    for day in [1]: #tqdm(range(365), desc='Processing days'):
        # try:
        start_row = day * 48
        end_row = (day + 1) * 48

        # 获取当日数据
        p_from_grid_day = p_from_grid_df.iloc[start_row:end_row]
        reactive_power_day = reactive_power_df.iloc[start_row:end_row]
        p_to_grid_day = p_to_grid_df.iloc[start_row:end_row]

        try:
            basic_earn_daily = Basic_earn(p_to_grid_day, p_from_grid_day)
        except Exception as e:
            basic_earn_daily = 0
            # 处理异常
            print(f"调用 Basic_earn 时出错：{e}")

        row_data = daily_EV_OPT.iloc[day]
        # 将Pandas Series转换为DataFrame
        row_data_df = pd.DataFrame(row_data.tolist())  # 每个列表成为一列
        row_data_df = row_data_df.T
        # 使用原始的Series索引作为DataFrame的列名
        row_data_df.columns = row_data.index

        # 分开正负部分
        positive_df = row_data_df.copy()
        negative_df = row_data_df.copy()

        # 分割正、负部分
        positive_df[positive_df < 0] = 0  # 负数设为0
        negative_df[negative_df > 0] = 0  # 正数设为0
        negative_df = np.abs(negative_df)  # 负数取绝对值

        # 备份原始索引
        original_index = p_from_grid_day.index

        # 在赋值前重置索引
        p_from_grid_day.reset_index(drop=True, inplace=True)  # 重置索引
        p_to_grid_day.reset_index(drop=True, inplace=True)

        # print("p_from_grid_day 索引:", p_from_grid_day.index.tolist())
        # print("positive_df 索引:", positive_df.index.tolist())
        # # 检查数据类型
        # print("p_from_grid_day 数据类型:", p_from_grid_day.dtypes)
        # print("positive_df 数据类型:", positive_df.dtypes)

        # 将数据更新到对应的数据框
        for node in daily_EV_OPT.columns:  # 第一列为 "Day"，忽略[1:]
            p_from_grid_day[node] = positive_df[node]  # 更新 p_from_grid_day
            p_to_grid_day[node] = negative_df[node]  # 更新 p_to_grid_day

        # 恢复原始索引
        p_from_grid_day.index = original_index  # 恢复索引
        p_to_grid_day.index = original_index

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
        # print(day)

# """
        opf = OPF(nodedata_dict, re_capacity_dict)

        # 创建文件夹路径
        output_file_path_day = f'annual_powerflow_{sen_type}_{strategy_type}/{ev_p}/{day}'
        if not os.path.exists(output_file_path_day):
            # 如果不存在，则创建文件夹
            os.makedirs(output_file_path_day)

        # try:
            # 调用外部模块的函数
        generator_costs, system_losses, import_powers, gen_powers, renewable_curtailed = opf.run_ts_opf(ev_p, v2g, output_file_path)
        # except Exception as e:
        #     # 处理异常
        #     print(f"{day}天调用 opf.run_ts_opf 时出错：{e}")
        #     # 设置默认值，确保代码继续运行
        #     generator_costs = []  # 赋予默认值
        #     system_losses = []
        #     import_powers = []
        #     gen_powers = []
        #     renewable_curtailed = []
        #     error_day.append(day)

        cost_import = np.sum(np.array(import_powers) * 0.5 * 1000 * C_tran)  # 0.5h MW

        # 将 generator_costs 转换为 DataFrame
        generator_all_data = []
        cost_gen = 0  # 初始化成本总和
        for period, costs in enumerate(generator_costs):
            for gen_index, gen_cost in costs:
                generator_all_data.append([period, gen_index, gen_cost])
                cost_gen += gen_cost  # 累加成本
        df_generator_costs = pd.DataFrame(generator_all_data, columns=['Period', 'Generator_Index', 'Cost'])

        cost_daily = cost_import/7.2 + cost_gen    # 买电成本+发电成本
        lose_daily = np.sum(np.array(system_losses))
        total_import_daily = np.sum(np.array(import_powers))
        total_gen_daily = np.sum(np.array(gen_powers))
        total_renewable_curtailed = np.sum(np.array(renewable_curtailed))

        # 定义一个辅助函数用于将数据保存到CSV文件

        # 将各个变量的值保存到CSV文件
        save_to_csv(df_generator_costs, output_file_path_day,f'generator_costs')
        save_to_csv(system_losses, output_file_path_day,f'system_losses')
        save_to_csv(import_powers, output_file_path_day,f'import_powers')

        annual_costs.append(cost_daily)
        annual_loses.append(lose_daily)
        annual_earns.append(basic_earn_daily)
        annual_imports.append(total_import_daily)
        annual_gens.append(total_gen_daily)
        annual_curtailed.append(total_renewable_curtailed)
        # print(f'第{day + 1}天的成本: {cost_daily}')
        # print(f'第{day + 1}天的网损: {lose_daily}')
        # print(f'第{day + 1}天的收入: {basic_earn_daily}')
        # print(f'第{day + 1}天的进口: {total_import_daily}')
        # print(f'第{day + 1}天的发电: {total_gen_daily}')
        # except Exception as e:
        #     annual_costs.append(0)
        #     annual_loses.append(0)
        #     annual_earns.append(0)
        #     annual_imports.append(0)
        #     annual_gens.append(0)
        #     annual_curtailed.append(0)
        #     error_day.append(day)
        #     print(f'处理第 {day} 天的数据时出错：{e}')

    # 创建DataFrame，将不同数据作为不同的列
    # annual_data = pd.DataFrame({
    #     'Day': range(1, len(annual_costs) + 1),
    #     'Costs': annual_costs,
    #     'Loses': annual_loses,
    #     'Earns': annual_earns,
    #     'Imports': annual_imports,
    #     'Generators': annual_gens,
    #     'curtail' : annual_curtailed
    # })

    # # 保存到同一个CSV文件
    # csv_file_path = f'{output_file_path}/{ev_p}.csv'
    # # 保存DataFrame到CSV文件
    # annual_data.to_csv(csv_file_path, index=False)
    #
    # print(f"数据已保存到 {csv_file_path}")
    # print(error_day)
    # # """


for ev_p in tqdm([0.15, 0.3, 0.5, 1], desc="loop over ev_p"):
    v2g = 0
    daily_powerflow(ev_p, v2g)
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


