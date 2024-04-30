import os
from tqdm import tqdm
from gridinfo import nodes, initial_EV, node_mapping,start_points,end_points
from docplex.mp.model import Model
# v2g order
# from anode_EVloadDOC_v2gorder import EVload_node
# order
# from anode_EVloadDOC_order import EVload_node
# import importlib
import pandas as pd
import numpy as np
# import math

# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #
k_values = {
    1.0: {'k1': -2.685e-11, 'k2': 1.539e-07, 'k3': -0.0003261, 'k4': 1},
    0.8: {'k1': -8.732e-12, 'k2': 6.271e-08, 'k3': -0.0001947, 'k4': 1},
    0.6: {'k1': -2.562e-12, 'k2': 2.665e-08, 'k3': -0.0001276, 'k4': 1},
    0.4: {'k1': -5.362e-13, 'k2': 9.537e-09, 'k3': -7.764e-05, 'k4': 1},
    0.3: {'k1': -3.084e-13, 'k2': 6.622e-09, 'k3': -6.492e-05, 'k4': 1},
    0.2: {'k1': -1.934e-13, 'k2': 4.866e-09, 'k3': -5.581e-05, 'k4': 1},
    0.1: {'k1': -1.292e-13, 'k2': 3.727e-09, 'k3': -4.894e-05, 'k4': 1}
}

# ev_p = 0.15
v2g = 0
daily_average = 24.4 #单程平均出行距离 12.2 公里
charging_delta = 0.11 #1/(70*0.6/0.188/24.4) 不要改这个

# 老化数据储存
aging_path = "vehicle_cycles/onlyWT_v2g"
# EV曲线储存
ev_power_path = 'annual_EV_onlyWT_v2g/'

def append_to_csv(data, filename):
    df = pd.DataFrame(data).transpose()  # 将向量转换为DataFrame的一行
    df.to_csv(filename, mode='a', header=False, index=False)  # 追加模式，不写入索引和表头

def assign_dod_to_nearest_level(dod):
    dod_levels = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    return min(dod_levels, key=lambda x: abs(x - dod))

def calculate_rc_dod(cycle_num, k_values, nearest_dod):#计算相对容量
    k1 = k_values[nearest_dod]['k1']
    k2 = k_values[nearest_dod]['k2']
    k3 = k_values[nearest_dod]['k3']
    k4 = k_values[nearest_dod]['k4']
    rc = k1*cycle_num**3 + k2*cycle_num**2 + k3*cycle_num + k4
    return rc


def calculate_cycle_num(RC, k_values, nearest_dod):
    k1 = k_values[nearest_dod]['k1']
    k2 = k_values[nearest_dod]['k2']
    k3 = k_values[nearest_dod]['k3']
    k4 = k_values[nearest_dod]['k4'] - RC

    # 系数equation
    coefficients = [k1, k2, k3, k4]
    # mumpy求解方程
    roots = np.roots(coefficients)
    #过滤正实数
    real_roots = [root.real for root in roots if root.imag == 0 and root.real >= 0]

    return min(real_roots)

class Node_annual:
    def __init__(self, node_num,ev_p):
        self.node_num = node_num
        self.ev_p = ev_p
        self.ev_all = int(initial_EV[node_mapping[node_num]])
        self.ev_ini = round(self.ev_all * ev_p)
        self.ev_num = round(self.ev_ini * charging_delta)

    def initialize_aging_dict(self):
        all_vehicle_ids = list(range(self.ev_ini))
        RC_values = [1] * self.ev_ini
        replace_counts = [0] * self.ev_ini

        return all_vehicle_ids, RC_values, replace_counts
    def read_ev(self):
        node_type = "community"
        # 读取车辆的离开和返回时间 weekday
        weekday_df = pd.read_csv(f'EV_KDE/{node_type}_node_{int(self.node_num)}_weekday.csv')
        weekday_ev_distribution = weekday_df['staying'].to_numpy()
        weekday_ev_leaving = weekday_df['leaving_next'].to_numpy()
        weekday_ev_arriving = weekday_df['arriving_next'].to_numpy()

        # self.slow_charging_distribution_weekday = np.round(weekday_ev_distribution * ev_p * charging_delta)
        slow_arriving_weekday = np.ceil(weekday_ev_arriving * self.ev_p * charging_delta)
        slow_leaving_weekday = np.floor(weekday_ev_leaving * self.ev_p * charging_delta)
        # print(slow_arriving_weekday)
        # print(slow_leaving_weekday)
        self.quick_charging_distribution_weekday = np.zeros(48)
        #
        # 初始化停留车辆数组
        self.slow_charging_distribution_weekday = np.zeros_like(slow_arriving_weekday, dtype=int)
        # 初始化第一个时间点
        self.slow_charging_distribution_weekday[0] = self.ev_num
        # 计算每个时间点的停留车辆数
        for i in range(1, len(self.slow_charging_distribution_weekday)):
            # 计算预期到达和离开的车辆数
            expected_arrivals = slow_arriving_weekday[i - 1]
            expected_departures = slow_leaving_weekday[i - 1]
            # 计算下一个时间点的预期停留车辆数
            next_staying = self.slow_charging_distribution_weekday[i - 1] + expected_arrivals - expected_departures
            # 如果预期停留车辆数超出最大数量，调整到达车辆数以防止超载
            if next_staying > self.ev_num:
                # 调整到达车辆数，以保证停留车辆数不超过最大容量
                expected_arrivals = self.ev_num - self.slow_charging_distribution_weekday[
                    i - 1] + expected_departures
                slow_arriving_weekday[i - 1] = max(0, expected_arrivals)  # 确保调整后到达车辆数不为负数
            # 如果预期停留车辆数小于0，调整离开车辆数以防止负数
            elif next_staying < 0:
                # 调整离开车辆数，以保证停留车辆数不小于0
                expected_departures = self.slow_charging_distribution_weekday[i - 1] + expected_arrivals
                slow_leaving_weekday[i - 1] = max(0, expected_departures)  # 确保调整后离开车辆数不为负数
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
        slow_arriving_weekend = np.ceil(weekend_ev_arriving * self.ev_p * charging_delta)
        slow_leaving_weekend = np.floor(weekend_ev_leaving * self.ev_p * charging_delta)
        self.quick_charging_distribution_weekend = np.zeros(48)
        # 初始化停留车辆数组
        self.slow_charging_distribution_weekend = np.zeros_like(slow_arriving_weekday, dtype=int)
        self.slow_charging_distribution_weekend[0] = self.ev_num  # 初始时刻的EV数量
        # 计算每个时间点的停留车辆数
        for i in range(1, len(self.slow_charging_distribution_weekend)):
            # 计算预期到达和离开的车辆数
            expected_arrivals = slow_arriving_weekend[i - 1]
            expected_departures = slow_leaving_weekend[i - 1]
            # 计算下一个时间点的预期停留车辆数
            next_staying = self.slow_charging_distribution_weekend[i - 1] + expected_arrivals - expected_departures
            # 如果预期停留车辆数超出最大数量，调整到达车辆数以防止超载
            if next_staying > self.ev_num:
                # 调整到达车辆数，以保证停留车辆数不超过最大容量
                expected_arrivals = self.ev_num - self.slow_charging_distribution_weekend[
                    i - 1] + expected_departures
                slow_arriving_weekend[i - 1] = max(0, expected_arrivals)  # 确保调整后到达车辆数不为负数
            # 如果预期停留车辆数小于0，调整离开车辆数以防止负数
            elif next_staying < 0:
                # 调整离开车辆数，以保证停留车辆数不小于0
                expected_departures = self.slow_charging_distribution_weekend[i - 1] + expected_arrivals
                slow_leaving_weekend[i - 1] = max(0, expected_departures)  # 确保调整后离开车辆数不为负数
            # 更新该时间点的停留车辆数
            self.slow_charging_distribution_weekend[i] = self.slow_charging_distribution_weekend[
                                                             i - 1] + expected_arrivals - expected_departures
            # # 重新确保该时间点的车辆数在合理范围内
            # self.slow_charging_distribution_weekend[i] = min(max(self.slow_charging_distribution_weekend[i], 0),
            #                                                  self.ev_num)
    def year_sim(self):
        self.read_ev()  # 读取csv
        all_vehicle_ids, RC_values, replace_counts = self.initialize_aging_dict() #初始化aging
        error_day = []
        ev_dir = ev_power_path + f'{self.ev_p}'
        # CSV文件的路径
        ev_file_path1 = f"{ev_dir}/{node_num}_charge_values.csv"
        ev_file_path2 = f"{ev_dir}/{node_num}_discharge_values.csv"
        vehicle_cycles_info = {1: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.3: 0, 0.2: 0, 0.1: 0}
        for day in range(365):
            #取出车的编号
            start_index = (day * self.ev_num) % len(all_vehicle_ids)
            end_index = start_index + self.ev_num
            if end_index > len(all_vehicle_ids):
                day_ids = all_vehicle_ids[start_index:] + all_vehicle_ids[:end_index % len(all_vehicle_ids)]
            else:
                day_ids = all_vehicle_ids[start_index:end_index]

            # 计算`day`除以7的余数
            remainder = day % 7
            # 如果余数是5或6，则这一天是
            if remainder == 5 or remainder == 6:
                slow_charging_distribution = self.slow_charging_distribution_weekend
            else:
                slow_charging_distribution = self.slow_charging_distribution_weekday


            # 读取CSV文件，指定没有标题行
            df_charge = pd.read_csv(ev_file_path1, header=None)
            # 读取CSV文件，指定没有标题行
            df_discharge = pd.read_csv(ev_file_path2, header=None)
            # 选择特定行，第day行
            charge_row = df_charge.iloc[day]  # 直接读取第day行
            charge_values = np.array(charge_row).astype(int)  # 转换为整数类型
            discharge_row = df_discharge.iloc[day]  # 直接读取第day行
            discharge_values = np.array(discharge_row).astype(int)  # 转换为整数类型

            a = sum(charge_values)
            b = sum(discharge_values)
            c = self.ev_num * 12
            # 存储计算结果到CSV文件

            #求解每辆车充电
            # 假设每辆车需要充电的净时隙数为12，即SOC_vector的和应为0.05*12
            net_charging_slots = 12
            slot_charge_increment = 0.05 #半个小时能充0.05SOC

            mdl = Model('EV_charging')

            # 为每辆车创建决策变量
            # 对于charging_vehicles_info中的每一辆车，创建两组二进制决策变量
            # 为每辆车创建决策变量
            charging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"charging_{vehicle_id}") for vehicle_id in
                             day_ids}
            discharging_vars = {vehicle_id: mdl.binary_var_list(48, name=f"discharging_{vehicle_id}") for vehicle_id in
                                day_ids}
            charging_slack_vars = {vehicle_id: mdl.binary_var_list(48, name=f"charging_slack_{vehicle_id}") for
                                   vehicle_id in day_ids}
            discharging_slack_vars = {vehicle_id: mdl.binary_var_list(48, name=f"discharging_slack_{vehicle_id}") for
                                      vehicle_id in day_ids}

            # 添加约束以确保在任何时刻，每辆车只能充电或放电，或者不做任何操作
            for vehicle_id in day_ids:
                for t in range(48):
                    # 充电和放电不能同时发生
                    mdl.add_constraint(charging_vars[vehicle_id][t] + discharging_vars[vehicle_id][t]
                                       + charging_slack_vars[vehicle_id][t] + discharging_slack_vars[vehicle_id][
                                           t] <= 1)

            # 2. 充电需求约束: 每辆车的净充电量总和需要达到特定值

            for vehicle_id in day_ids:
                charge_sum = mdl.sum(charging_vars[vehicle_id])
                discharge_sum = mdl.sum(discharging_vars[vehicle_id])
                charge_sum_slack = mdl.sum(charging_slack_vars[vehicle_id])
                discharge_sum_slack = mdl.sum(discharging_slack_vars[vehicle_id])
                mdl.add_constraint(
                    charge_sum + charge_sum_slack - discharge_sum - discharge_sum_slack == net_charging_slots)

            # 3. 总体充电和放电约束
            for t in range(48):
                vehicle_discharge = mdl.sum(discharging_vars[vehicle_id][t] for vehicle_id in day_ids)
                vehicle_charge = mdl.sum(charging_vars[vehicle_id][t] for vehicle_id in day_ids)
                vehicle_charge_slack = mdl.sum(charging_slack_vars[vehicle_id][t] for vehicle_id in day_ids)
                vehicle_discharge_slack = mdl.sum(
                    discharging_slack_vars[vehicle_id][t] for vehicle_id in day_ids)

                mdl.add_constraint(vehicle_charge + vehicle_charge_slack >= charge_values[t])
                mdl.add_constraint(vehicle_discharge + vehicle_discharge_slack >= discharge_values[t])
                mdl.add_constraint(vehicle_charge_slack + vehicle_charge - vehicle_discharge - vehicle_discharge_slack
                                   == charge_values[t] - discharge_values[t])
                mdl.add_constraint(vehicle_charge_slack + vehicle_charge + vehicle_discharge + vehicle_discharge_slack
                                   <= slow_charging_distribution[t])

            # 创建一个列表，包含所有车辆的所有二进制变量
            all_charging_vars = [
                var
                for vehicle_vars in charging_vars.values()
                for var in vehicle_vars
            ]
            all_discharging_vars = [
                var
                for vehicle_vars in discharging_vars.values()
                for var in vehicle_vars
            ]
            all_charging_slack_vars = [
                var
                for vehicle_vars in charging_slack_vars.values()
                for var in vehicle_vars
            ]
            all_discharging_slack_vars = [
                var
                for vehicle_vars in discharging_slack_vars.values()
                for var in vehicle_vars
            ]
            weight = 1.5
            mdl.minimize(mdl.sum(all_charging_vars) + mdl.sum(all_discharging_vars)
                         + weight * (mdl.sum(all_charging_slack_vars) + mdl.sum(all_discharging_slack_vars)))
            # 求解模型
            solution = mdl.solve()

            # 打印解决方案
            if solution:
                # print("充电循环计算结束")
                # 分析每辆车的充放电活动
                for vehicle_id in day_ids:
                    prev_state = - 1  # 前一个时刻的充放电状态，初始化为0（无活动）
                    current_cycle_charge = - 0.6  # 当前充放电周期的累计SOC
                    current_soc = 0.3  # 当前SOC
                    inflection_soc = [0.9]  # 拐点SOC
                    cycle_sign = 0  #

                    for t in range(48):
                        charging_state = solution.get_value(charging_vars[vehicle_id][t]) + solution.get_value(
                            charging_slack_vars[vehicle_id][t])
                        discharging_state = solution.get_value(discharging_vars[vehicle_id][t]) + solution.get_value(
                            discharging_slack_vars[vehicle_id][t])
                        current_state = charging_state - discharging_state  # 当前时刻的充放电状态 1 0 -1
                        soc_change = slot_charge_increment * current_state # DELTA SOC
                        current_cycle_charge += soc_change # 当前充放电周期的累计SOC
                        current_soc += soc_change #当前SOC

                        # 检查是否终点 总是将最后一个状态视为新的拐点
                        if t == 47:
                            cycle_sign += 1
                            inflection_soc.append(current_soc)  # 添加最后的SOC作为新的拐点
                        else:
                            # 检测充放电状态的改变
                            if current_state * prev_state < 0: # 充电状态改变
                                cycle_sign += 1
                                inflection_soc.append(current_soc)
                        if cycle_sign > 1:
                            dod = max(inflection_soc) - min(inflection_soc) # 计算DOD
                            nearest_dod = assign_dod_to_nearest_level(dod)
                            vehicle_cycles_info[nearest_dod] += 1 #给相应的cycle+1

                            # 计算RC
                            current_relative_capacity = RC_values[vehicle_id]
                            cycle_num = calculate_cycle_num(current_relative_capacity, k_values, nearest_dod)
                            cycle_number = cycle_num + 1 #当前循环次数
                            new_relative_capacity = calculate_rc_dod(cycle_number, k_values, nearest_dod)

                            if new_relative_capacity < 0.8:
                                replace_counts[vehicle_id] += 1  # 重置电池
                                RC_values[vehicle_id] = 1  # 重置
                            else:
                                RC_values[vehicle_id] = new_relative_capacity

                            inflection_soc = [current_soc]  # 重置当周期记录

                        if current_state != 0:
                            prev_state = current_state  # 更新前一个时刻的状态
                        else:
                            prev_state = prev_state

            else:
                # 获取求解器状态
                status = mdl.get_solve_status()
                print(f"{self.node_num}{day}Solver status:", status)
                error_day.append([self.node_num,day])


            # # 快充
            # cycle_quick = EV_charging_quick
            # vehicle_cycles_info[0.6] += cycle_quick

        EV_data = {
            "Vehicle_ID": all_vehicle_ids,
            "RC_Values": RC_values,
            "Replace_Counts": replace_counts
        }
        EV_df = pd.DataFrame(EV_data)

        # Construct the directory and file paths
        directory_path_ev = os.path.join(aging_path, str(self.ev_p), "EV_battery")
        file_name = f"EV_{self.node_num}.csv"
        full_file_path = os.path.join(directory_path_ev, file_name)
        # 不存在则建立
        os.makedirs(directory_path_ev, exist_ok=True)

        # Save the DataFrame to CSV
        EV_df.to_csv(full_file_path, index=False)

        # 转换为数据框
        df_charge = pd.DataFrame(list(vehicle_cycles_info.items()), columns=['Parameter', 'Value'])
        # 定义文件路径
        aging_folder_path = os.path.join(aging_path, str(self.ev_p))
        if not os.path.exists(aging_folder_path):
            # 如果不存在，创建文件夹
            os.makedirs(aging_folder_path)
        file_name = os.path.join(aging_folder_path, f"{self.node_num}.csv")
        # 保存数据框为 CSV 文件
        df_charge.to_csv(file_name, index=False)
        if error_day:
            # 将 error_day 转换为 DataFrame，并设置列名
            df_error_day = pd.DataFrame(error_day, columns=["node_num", "day"])
            # 保存到 CSV 文件
            file_name_error = os.path.join(aging_folder_path, f"{self.node_num}_error.csv")
            df_error_day.to_csv(file_name_error, index=False)  # 示例：将其保存为 CSV 文件

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

