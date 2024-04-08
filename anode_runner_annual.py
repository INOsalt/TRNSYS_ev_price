import os
from tqdm import tqdm
from gridinfo import nodes
from anode_EVloadDOC_v2gorder import EVload_node
from docplex.mp.model import Model
import importlib
import pandas as pd
import numpy as np
import math

# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #

node_num = 303
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

# 创建初始向量
EV_initial_capacity = np.full(120, 70)  # 每辆EV的初始容量
EV_initial_SOC = np.full(120, 0.9)  # 每辆EV的初始SOC

# 定义路径
daily_mileages_path = 'daily_vehicle_mileages.csv'

# 读取每日里程文件
daily_mileages = pd.read_csv(daily_mileages_path)


def assign_dod_to_nearest_level(dod):
    dod_levels = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    return min(dod_levels, key=lambda x: abs(x - dod))

for day in range(365):

    # 初始化记录每辆车充放电循环信息的字典
    vehicle_cycles_info = {
        vehicle_id: {1: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.3: 0, 0.2: 0, 0.1: 0} for vehicle_id in range(120)
    }

    # 初始化
    charging_vehicles_per_timeslot = np.zeros(48)
    leaving_vehicles_per_timeslot = np.zeros(48)
    arriving_vehicles_per_timeslot = np.zeros(48)
    # 初始化记录需要充电的车辆信息的字典
    charging_vehicles_info = {}

    # 对于每辆车
    for vehicle_id in range(120):
        # 读取车辆的离开和返回时间
        times = pd.read_csv(f'EVs_new/{vehicle_id}.csv', header=None).iloc[day]
        departure_time, return_time = int(times[0]), int(times[1])

        # 计算能耗
        energy_consumed = daily_mileages.iloc[day, vehicle_id] * 0.188
        # 更新SOC
        EV_initial_SOC[vehicle_id] -= (energy_consumed / EV_initial_capacity[vehicle_id])

        # 检查是否需要充电
        if EV_initial_SOC[vehicle_id] <= 0.3:
            # 如果需要充电，记录车辆编号和离开/返回时间
            charging_vehicles_info[vehicle_id] = (departure_time, return_time)
            # 计算充电的净时隙数，四舍五入到最接近的整数
            net_charging = (0.9 - 0.3) * EV_initial_capacity[vehicle_id]
            net_charging_slots = math.ceil(net_charging / 7 * 2)
            # 使用NumPy来优化时间段的更新
            charging_vehicles_per_timeslot[:departure_time] += 1
            charging_vehicles_per_timeslot[return_time:] += 1
            if departure_time > 0:
                leaving_vehicles_per_timeslot[departure_time - 1] += 1
            for t in range(47):
                arriving_vehicles_per_timeslot[t] = (charging_vehicles_per_timeslot[t + 1]
                                                     - charging_vehicles_per_timeslot[t]
                                                     + leaving_vehicles_per_timeslot[t])
            arriving_vehicles_per_timeslot[47] = 0

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

    slow_charging_distribution = charging_vehicles_per_timeslot
    quick_charging_distribution = [0] * 48
    slow_arriving = arriving_vehicles_per_timeslot
    slow_leaving = leaving_vehicles_per_timeslot

    (mic_EV_load_slow, mic_EV_load_quick, node_P_total, node_P_basic_and_EV, node_slack_load,
     P_basic_dict, charge_values, discharge_values) \
        = EVload_node(slow_charging_distribution, quick_charging_distribution,slow_arriving,slow_leaving,
                ev_p,v2g,data1_dir,nodedata_dict, re_capacity_dict,node_num)

    # 存储计算结果到CSV文件
    def append_to_csv(data, filename):
        df = pd.DataFrame(data).transpose()  # 将向量转换为DataFrame的一行
        df.to_csv(filename, mode='a', header=False, index=False)  # 追加模式，不写入索引和表头


    # 追加node_P_total和node_P_basic_and_EV到对应的文件
    append_to_csv(node_P_total, f'{data1_dir}/node_P_total.csv')
    append_to_csv(node_P_basic_and_EV, f'{data1_dir}/node_P_basic_and_EV.csv')

    #求解每辆车充电
    # 假设每辆车需要充电的净时隙数为12，即SOC_vector的和应为0.05*12
    net_charging_slots = 12
    slot_charge_increment = 0.05

    mdl = Model('EV_charging')

    # 为每辆车创建决策变量
    # 对于charging_vehicles_info中的每一辆车，创建两组二进制决策变量
    charging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"charging_{vehicle_id}") for vehicle_id in
                     charging_vehicles_info}
    discharging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"discharging_{vehicle_id}") for vehicle_id in
                        charging_vehicles_info}

    # 添加约束以确保在任何时刻，每辆车只能充电或放电，或者不做任何操作
    for vehicle_id in charging_vehicles_info:
        for t in range(48):
            # 充电和放电不能同时发生
            mdl.add_constraint(charging_vars[vehicle_id][t] + discharging_vars[vehicle_id][t] <= 1)

    # 添加约束
    for vehicle_id, (departure_time, return_time) in charging_vehicles_info.items():
        # 1. 充电时间约束: 在车辆离开和返回时间内，SOC变化量应为0
        for t in range(departure_time, return_time):
            mdl.add_constraint(charging_vars[vehicle_id][t] == 0)
            mdl.add_constraint(discharging_vars[vehicle_id][t] == 0)

        # 2. 充电需求约束: 每辆车的净充电量总和需要达到特定值
        charge_sum = mdl.sum(charging_vars[vehicle_id])
        discharge_sum = mdl.sum(discharging_vars[vehicle_id])
        mdl.add_constraint(charge_sum - discharge_sum == net_charging_slots)

    # 3. 总体充电和放电约束
    for t in range(48):
        mdl.add_constraint(
            mdl.sum(charging_vars[vehicle_id][t] for vehicle_id in charging_vehicles_info) == charge_values[t])
    # 总体放电约束
    for t in range(48):
        mdl.add_constraint(
            mdl.sum(discharging_vars[vehicle_id][t] for vehicle_id in charging_vehicles_info) == discharge_values[t])

    # 求解模型
    solution = mdl.solve()

    # 打印解决方案
    if solution:
        # 分析每辆车的充放电活动
        for vehicle_id in charging_vehicles_info:
            prev_state = 0  # 前一个时刻的充放电状态，初始化为0（无活动）
            current_cycle_charge = 0  # 当前充放电周期的累计充电量

            for t in range(48):
                charging_state = solution.get_value(charging_vars[vehicle_id][t])
                discharging_state = solution.get_value(discharging_vars[vehicle_id][t])
                current_state = charging_state - discharging_state  # 当前时刻的充放电状态
                soc_change = slot_charge_increment * current_state
                current_cycle_charge += soc_change

                # 检测充放电状态的改变
                if current_state != prev_state:
                    # 如果之前有充放电活动，且当前时刻的状态与前一时刻不同，结束一个周期
                    if abs(current_cycle_charge - soc_change) > 0:
                        dod = abs(current_cycle_charge - soc_change)  # 计算DOD
                        nearest_dod = assign_dod_to_nearest_level(dod)
                        vehicle_cycles_info[vehicle_id][nearest_dod] += 1

                    current_cycle_charge = soc_change  # 重置当前周期的累计充电量

                prev_state = current_state  # 更新前一个时刻的状态

            # 在分析结束前加上一次0.6的放电
            if current_cycle_charge != 0:  # 如果当前有正在进行的周期，则先结束这个周期
                dod = abs(current_cycle_charge)
                nearest_dod = assign_dod_to_nearest_level(dod)
                vehicle_cycles_info[vehicle_id][nearest_dod] += 1

            # 现在考虑额外的0.6放电
            vehicle_cycles_info[vehicle_id][0.6] += 1  # 加上一次0.6的放电

