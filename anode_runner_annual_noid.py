import os
from tqdm import tqdm
from gridinfo import nodes, initial_EV, node_mapping,start_points,end_points
from docplex.mp.model import Model
# v2g order
# from anode_EVloadDOC_v2gorder import EVload_node
# order
from anode_EVloadDOC_order import EVload_node
# import importlib
import pandas as pd
import numpy as np
import math

# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #


# ev_p = 0.15
v2g = 0
daily_average = 24.4 #单程平均出行距离 12.2 公里
charging_delta = 0.11 #1/(70*0.6/0.188/24.4)

# 老化数据储存
aging_path = "vehicle_cycles/onlyWT_order"
# 功率曲线储存
power_path = 'data_annual_onlyWT_order/'
# 读取CSV文件
folder_path = 'annual_onlyWT/'
p_from_grid_filename = folder_path + 'P_from_grid_kW_total.csv'
reactive_power_filename = folder_path + 'reactive_power_total.csv'
p_to_grid_filename = folder_path + 'P_to_grid_kW_total.csv'

p_from_grid_df = pd.read_csv(p_from_grid_filename, dtype={'my_column': float})
reactive_power_df = pd.read_csv(reactive_power_filename, dtype={'my_column': float})
p_to_grid_df = pd.read_csv(p_to_grid_filename, dtype={'my_column': float})

# nodes = list(p_from_grid_df.columns)  # 假设列名为节点编号

# # 定义路径
# daily_mileages_path = 'daily_vehicle_mileages.csv'
# # 读取每日里程文件
# daily_mileages = pd.read_csv(daily_mileages_path)

def append_to_csv(data, filename):
    df = pd.DataFrame(data).transpose()  # 将向量转换为DataFrame的一行
    df.to_csv(filename, mode='a', header=False, index=False)  # 追加模式，不写入索引和表头

def assign_dod_to_nearest_level(dod):
    dod_levels = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    return min(dod_levels, key=lambda x: abs(x - dod))
class Node_annual:
    def __init__(self, node_num,ev_p):
        self.node_num = node_num
        self.ev_ini = initial_EV[node_mapping[node_num]]
        self.ev_p = ev_p
        self.ev_num = round(self.ev_ini * ev_p * charging_delta)

    def read_ev(self):

        if self.node_num in end_points:
            node_type = "work"
            # 读取车辆的离开和返回时间 weekday
            weekday_df = pd.read_csv(f'EV_KDE/{node_type}_node_{int(self.node_num)}_weekday.csv')
            weekday_ev_distribution = weekday_df['staying'].to_numpy()
            weekday_ev_leaving = weekday_df['leaving_next'].to_numpy()
            weekday_ev_arriving = weekday_df['arriving_next'].to_numpy()

            self.slow_charging_distribution_weekday = np.round(weekday_ev_distribution * self.ev_p * charging_delta)
            self.slow_arriving_weekday = np.round(weekday_ev_arriving * self.ev_p * charging_delta)
            self.slow_leaving_weekday = np.round(weekday_ev_leaving * self.ev_p * charging_delta)
            self.quick_charging_distribution_weekday = np.zeros(48)

            self.slow_charging_distribution_weekend = np.zeros(48)
            self.slow_arriving_weekend = np.zeros(48)
            self.slow_leaving_weekend = np.zeros(48)
            self.quick_charging_distribution_weekend = np.zeros(48)

        else:
            node_type = "community"
            # 读取车辆的离开和返回时间 weekday
            weekday_df = pd.read_csv(f'EV_KDE/{node_type}_node_{int(self.node_num)}_weekday.csv')
            weekday_ev_distribution = weekday_df['staying'].to_numpy()
            weekday_ev_leaving = weekday_df['leaving_next'].to_numpy()
            weekday_ev_arriving = weekday_df['arriving_next'].to_numpy()

            # self.slow_charging_distribution_weekday = np.round(weekday_ev_distribution * ev_p * charging_delta)
            self.slow_arriving_weekday = np.round(weekday_ev_arriving * self.ev_p * charging_delta)
            self.slow_leaving_weekday = np.round(weekday_ev_leaving * self.ev_p * charging_delta)
            self.quick_charging_distribution_weekday = np.zeros(48)
            #
            # 初始化停留车辆数组
            self.slow_charging_distribution_weekday = np.zeros_like(self.slow_arriving_weekday, dtype=int)
            # 初始化第一个时间点
            self.slow_charging_distribution_weekday[0] = self.ev_num
            # 计算每个时间点的停留车辆数
            for i in range(1, len(self.slow_charging_distribution_weekday)):
                # 计算预期到达和离开的车辆数
                expected_arrivals = self.slow_arriving_weekday[i - 1]
                expected_departures = self.slow_leaving_weekday[i - 1]
                # 计算下一个时间点的预期停留车辆数
                next_staying = self.slow_charging_distribution_weekday[i - 1] + expected_arrivals - expected_departures
                # 如果预期停留车辆数超出最大数量，调整到达车辆数以防止超载
                if next_staying > self.ev_num:
                    # 调整到达车辆数，以保证停留车辆数不超过最大容量
                    expected_arrivals = self.ev_num - self.slow_charging_distribution_weekday[
                        i - 1] + expected_departures
                    self.slow_arriving_weekday[i - 1] = max(0, expected_arrivals)  # 确保调整后到达车辆数不为负数
                # 如果预期停留车辆数小于0，调整离开车辆数以防止负数
                elif next_staying < 0:
                    # 调整离开车辆数，以保证停留车辆数不小于0
                    expected_departures = self.slow_charging_distribution_weekday[i - 1] + expected_arrivals
                    self.slow_leaving_weekday[i - 1] = max(0, expected_departures)  # 确保调整后离开车辆数不为负数
                # 更新该时间点的停留车辆数
                self.slow_charging_distribution_weekday[i] = self.slow_charging_distribution_weekday[
                                                                 i - 1] + expected_arrivals - expected_departures
                # 重新确保该时间点的车辆数在合理范围内
                self.slow_charging_distribution_weekday[i] = min(max(self.slow_charging_distribution_weekday[i], 0),
                                                                 self.ev_num)

            # 读取车辆的离开和返回时间 weekend
            weekend_df = pd.read_csv(f'EV_KDE/{node_type}_node_{int(self.node_num)}_weekend.csv')
            weekend_ev_distribution = weekend_df['staying'].to_numpy()
            weekend_ev_leaving = weekend_df['leaving_next'].to_numpy()
            weekend_ev_arriving = weekend_df['arriving_next'].to_numpy()

            # self.slow_charging_distribution_weekend = np.round(weekend_ev_distribution * ev_p * charging_delta)
            self.slow_arriving_weekend = np.round(weekend_ev_arriving * self.ev_p * charging_delta)
            self.slow_leaving_weekend = np.round(weekend_ev_leaving * self.ev_p * charging_delta)
            self.quick_charging_distribution_weekend = np.zeros(48)
            # 初始化停留车辆数组
            self.slow_charging_distribution_weekend = np.zeros_like(self.slow_arriving_weekday, dtype=int)
            self.slow_charging_distribution_weekend[0] = self.ev_num  # 初始时刻的EV数量
            # 计算每个时间点的停留车辆数
            for i in range(1, len(self.slow_charging_distribution_weekend)):
                # 计算预期到达和离开的车辆数
                expected_arrivals = self.slow_arriving_weekend[i - 1]
                expected_departures = self.slow_leaving_weekend[i - 1]
                # 计算下一个时间点的预期停留车辆数
                next_staying = self.slow_charging_distribution_weekend[i - 1] + expected_arrivals - expected_departures
                # 如果预期停留车辆数超出最大数量，调整到达车辆数以防止超载
                if next_staying > self.ev_num:
                    # 调整到达车辆数，以保证停留车辆数不超过最大容量
                    expected_arrivals = self.ev_num - self.slow_charging_distribution_weekend[
                        i - 1] + expected_departures
                    self.slow_arriving_weekend[i - 1] = max(0, expected_arrivals)  # 确保调整后到达车辆数不为负数
                # 如果预期停留车辆数小于0，调整离开车辆数以防止负数
                elif next_staying < 0:
                    # 调整离开车辆数，以保证停留车辆数不小于0
                    expected_departures = self.slow_charging_distribution_weekend[i - 1] + expected_arrivals
                    self.slow_leaving_weekend[i - 1] = max(0, expected_departures)  # 确保调整后离开车辆数不为负数
                # 更新该时间点的停留车辆数
                self.slow_charging_distribution_weekend[i] = self.slow_charging_distribution_weekend[
                                                                 i - 1] + expected_arrivals - expected_departures
                # 重新确保该时间点的车辆数在合理范围内
                self.slow_charging_distribution_weekend[i] = min(max(self.slow_charging_distribution_weekend[i], 0),
                                                                 self.ev_num)


    def year_sim(self):
        self.read_ev() #读取csv
        data1_dir = power_path + f'{self.ev_p}'
        if not os.path.exists(data1_dir):
            # 如果不存在，创建文件夹
            os.makedirs(data1_dir)
        vehicle_cycles_info = {1: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.3: 0, 0.2: 0, 0.1: 0}
        for day in range(365):
            # 计算`day`除以7的余数
            remainder = day % 7
            # 如果余数是5或6，则这一天是
            if remainder == 5 or remainder == 6:
                slow_charging_distribution = self.slow_charging_distribution_weekend
                quick_charging_distribution = self.quick_charging_distribution_weekend
                slow_arriving = self.slow_arriving_weekend
                slow_leaving = self.slow_leaving_weekend
            else:
                slow_charging_distribution = self.slow_charging_distribution_weekday
                quick_charging_distribution = self.quick_charging_distribution_weekday
                slow_arriving = self.slow_arriving_weekday
                slow_leaving = self.slow_leaving_weekday

            EV_charging_slow = np.max(slow_charging_distribution)
            EV_charging_quick = np.sum(quick_charging_distribution) / 2

            # 初始化记录每辆车充放电循环信息的字典
            # vehicle_cycles_info = {
            #     vehicle_id: {1: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.3: 0, 0.2: 0, 0.1: 0} for vehicle_id in range(EV_num)周末
            # }
            # 初始化
            # 初始化记录需要充电的车辆信息的字典

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

            (mic_EV_load_slow, mic_EV_load_quick, node_P_total, node_P_basic_and_EV, node_slack_load,
             P_basic_dict, charge_values, discharge_values) \
                = EVload_node(slow_charging_distribution, quick_charging_distribution,slow_arriving,slow_leaving,
                        self.ev_p,v2g,data1_dir,nodedata_dict, re_capacity_dict,self.node_num,self.ev_num)

            a = sum(charge_values)
            b = sum(discharge_values)
            c = self.ev_num * 12
            # 存储计算结果到CSV文件

            charge_values = np.round(np.array(charge_values))
            discharge_values = np.round(np.array(discharge_values))

            # 追加node_P_total和node_P_basic_and_EV到对应的文件
            append_to_csv(node_P_total, f'{data1_dir}/{self.node_num}_P_total.csv')
            append_to_csv(node_P_basic_and_EV, f'{data1_dir}/{self.node_num}_P_basic_and_EV.csv')

            #求解每辆车充电
            # 假设每辆车需要充电的净时隙数为12，即SOC_vector的和应为0.05*12
            net_charging_slots = 12
            slot_charge_increment = 0.05 #半个小时能充0.05SOC

            mdl = Model('EV_charging')

            # 为每辆车创建决策变量
            # 对于charging_vehicles_info中的每一辆车，创建两组二进制决策变量
            # 为每辆车创建决策变量
            charging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"charging_{vehicle_id}") for vehicle_id in
                             range(self.ev_num)}
            discharging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"discharging_{vehicle_id}") for vehicle_id in
                                range(self.ev_num)}

            # 添加约束以确保在任何时刻，每辆车只能充电或放电，或者不做任何操作
            for vehicle_id in range(self.ev_num):
                for t in range(48):
                    # 充电和放电不能同时发生
                    mdl.add_constraint(charging_vars[vehicle_id][t] + discharging_vars[vehicle_id][t] <= 1)

            # # 添加约束
            # for vehicle_id, (departure_time, return_time) in charging_vehicles_info_slow.items():
            #     # 1. 充电时间约束: 在车辆离开和返回时间内，SOC变化量应为0
            #     for t in range(departure_time, return_time):
            #         mdl.add_constraint(charging_vars[vehicle_id][t] == 0)
            #         mdl.add_constraint(discharging_vars[vehicle_id][t] == 0)

            # 2. 充电需求约束: 每辆车的净充电量总和需要达到特定值
            total_charge_sum = mdl.sum(mdl.sum(charging_vars[vehicle_id]) for vehicle_id in range(self.ev_num))
            total_discharge_sum = mdl.sum(
                mdl.sum(discharging_vars[vehicle_id]) for vehicle_id in range(self.ev_num))
            mdl.add_constraint(
                total_charge_sum - total_discharge_sum == net_charging_slots * self.ev_num)
            for vehicle_id in range(self.ev_num):
                mdl.add_constraint(mdl.sum(charging_vars[vehicle_id]) >= 4)
                # # 2. 充电需求约束: 每辆车的净充电量总和需要达到特定值
                # charge_sum = mdl.sum(charging_vars[vehicle_id])
                # discharge_sum = mdl.sum(discharging_vars[vehicle_id])
                # mdl.add_constraint(charge_sum - discharge_sum == net_charging_slots)

            # 3. 总体充电和放电约束
            for t in range(48):
                mdl.add_constraint(
                    mdl.sum(charging_vars[vehicle_id][t] for vehicle_id in range(self.ev_num)) == charge_values[t])
            # 总体放电约束
            for t in range(48):
                mdl.add_constraint(
                    mdl.sum(discharging_vars[vehicle_id][t] for vehicle_id in range(self.ev_num)) == discharge_values[t])

            # 求解模型
            solution = mdl.solve()

            # 打印解决方案
            if solution:
                # print("充电循环计算结束")
                # 分析每辆车的充放电活动
                for vehicle_id in range(math.ceil(EV_charging_slow)):
                    prev_state = - 1  # 前一个时刻的充放电状态，初始化为0（无活动）
                    current_cycle_charge = - 0.6  # 当前充放电周期的累计SOC
                    current_soc = 0.3 #当前SOC
                    inflection_soc = [0.9] #拐点SOC
                    cycle_sign = 0 #

                    for t in range(48):
                        charging_state = solution.get_value(charging_vars[vehicle_id][t])
                        discharging_state = solution.get_value(discharging_vars[vehicle_id][t])
                        current_state = charging_state - discharging_state  # 当前时刻的充放电状态 1 0 -1
                        soc_change = slot_charge_increment * current_state # DELTA SOC
                        current_cycle_charge += soc_change # 当前充放电周期的累计SOC
                        current_soc += soc_change #当前SOC

                        # 检测充放电状态的改变
                        if current_state * prev_state < 0: # 充电状态改变
                            cycle_sign += 1
                            inflection_soc.append(current_soc)
                            if cycle_sign > 1:
                                dod = max(inflection_soc) - min(inflection_soc) # 计算DOD
                                nearest_dod = assign_dod_to_nearest_level(dod)
                                vehicle_cycles_info[nearest_dod] += 1 #给相应的cycle+1

                                inflection_soc = [current_soc]  # 重置当周期记录

                        if current_state != 0:
                            prev_state = current_state  # 更新前一个时刻的状态
                        else:
                            prev_state = prev_state
            else:
                # 获取求解器状态
                status = mdl.get_solve_status()
                print(f"{self.node_num}Solver status:", status)
            # 快充
            cycle_quick = EV_charging_quick
            vehicle_cycles_info[0.6] += cycle_quick

        # 转换为数据框
        df = pd.DataFrame(list(vehicle_cycles_info.items()), columns=['Parameter', 'Value'])
        # 定义文件路径
        aging_folder_path = os.path.join(aging_path, str(self.ev_p))
        if not os.path.exists(aging_folder_path):
            # 如果不存在，创建文件夹
            os.makedirs(aging_folder_path)
        file_name = os.path.join(aging_folder_path, f"{self.node_num}.csv")
        # 保存数据框为 CSV 文件
        df.to_csv(file_name, index=False)
        print("已保存文件:", file_name)


# # for node_num in [201, 202, 203, 204, 205, 206, 207, 208, 209, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
# #                   312, 313, 314, 315, 316, 317, 318, 401, 402, 403, 404, 405, 407]:
# # for node_num in [204,209,302,304,307,317]:
# ev_p = 0.15
for ev_p in tqdm([0.15, 0.3, 0.5, 1], desc="Outer loop over ev_p"):
    for node_num in tqdm(start_points, desc="Inner loop over nodes", leave=False):
        sim = Node_annual(node_num, ev_p)
        sim.year_sim()
# ev_p = 1
# for node_num in tqdm([202, 203, 204, 205, 206, 207, 208, 209, 301, 302,
#                       303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
#                       313, 314, 315, 316, 317, 318, 401, 402, 403, 404,
#                       405, 407], desc="Inner loop over nodes", leave=False):
#     sim = Node_annual(node_num, ev_p)
#     sim.year_sim()



# sim = Node_annual(202)
# sim.year_sim()
# [202,205,303,307,311,405,407]

