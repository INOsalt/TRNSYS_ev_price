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
aging_path = "vehicle_cycles/onlyPV_v2g"
# EV曲线储存
ev_power_path = 'annual_EV_onlyPV_v2g/'

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
    def year_sim(self,day):
        error_day = []
        ev_dir = ev_power_path + f'{self.ev_p}'
        # CSV文件的路径
        ev_file_path1 = f"{ev_dir}/{node_num}_charge_values.csv"
        ev_file_path2 = f"{ev_dir}/{node_num}_discharge_values.csv"
        vehicle_cycles_info = {1: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.3: 0, 0.2: 0, 0.1: 0}
        # for day in range(365):
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
        print(charge_values)
        print(discharge_values)
        print(a,b,c)

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
        charging_slack_vars = {vehicle_id: mdl.binary_var_list(48, name=f"charging_slack_{vehicle_id}") for vehicle_id in
                         range(self.ev_num)}
        discharging_slack_vars = {vehicle_id: mdl.binary_var_list(48, name=f"discharging_slack_{vehicle_id}") for vehicle_id in
                               range(self.ev_num)}

        # slack
        # 添加约束以确保在任何时刻，每辆车只能充电或放电，或者不做任何操作
        for vehicle_id in range(self.ev_num):
            for t in range(48):
                # 充电和放电不能同时发生
                mdl.add_constraint(charging_vars[vehicle_id][t] + discharging_vars[vehicle_id][t]
                                   + charging_slack_vars[vehicle_id][t] + discharging_slack_vars[vehicle_id][t]<= 1)

        # # 添加约束
        # for vehicle_id, (departure_time, return_time) in charging_vehicles_info_slow.items():
        #     # 1. 充电时间约束: 在车辆离开和返回时间内，SOC变化量应为0
        #     for t in range(departure_time, return_time):
        #         mdl.add_constraint(charging_vars[vehicle_id][t] == 0)
        #         mdl.add_constraint(discharging_vars[vehicle_id][t] == 0)

        # 2. 充电需求约束: 每辆车的净充电量总和需要达到特定值
        # total_charge_sum = mdl.sum(mdl.sum(charging_vars[vehicle_id]) for vehicle_id in range(self.ev_num))
        # total_discharge_sum = mdl.sum(
        #     mdl.sum(discharging_vars[vehicle_id]) for vehicle_id in range(self.ev_num))
        # mdl.add_constraint(
        #     total_charge_sum - total_discharge_sum == net_charging_slots * self.ev_num)
        for vehicle_id in range(self.ev_num):
            charge_sum = mdl.sum(charging_vars[vehicle_id])
            discharge_sum = mdl.sum(discharging_vars[vehicle_id])
            charge_sum_slack = mdl.sum(charging_slack_vars[vehicle_id])
            discharge_sum_slack = mdl.sum(discharging_slack_vars[vehicle_id])
            mdl.add_constraint(charge_sum + charge_sum_slack - discharge_sum - discharge_sum_slack == net_charging_slots)

        # 3. 总体充电和放电约束
        for t in range(48):
            vehicle_discharge = mdl.sum(discharging_vars[vehicle_id][t] for vehicle_id in range(self.ev_num))
            vehicle_charge = mdl.sum(charging_vars[vehicle_id][t] for vehicle_id in range(self.ev_num))
            vehicle_charge_slack = mdl.sum(charging_slack_vars[vehicle_id][t] for vehicle_id in range(self.ev_num))
            vehicle_discharge_slack = mdl.sum(discharging_slack_vars[vehicle_id][t] for vehicle_id in range(self.ev_num))

            mdl.add_constraint(vehicle_charge + vehicle_charge_slack >= charge_values[t])
            mdl.add_constraint(vehicle_discharge + vehicle_discharge_slack >= discharge_values[t])
            mdl.add_constraint(vehicle_charge_slack + vehicle_charge - vehicle_discharge - vehicle_discharge_slack
                               == charge_values[t] - discharge_values[t])

        # 创建一个列表，包含所有车辆的所有二进制变量
        all_charging_slack_vars = [
            var
            for vehicle_vars in charging_slack_vars.values()
            for var in vehicle_vars
        ]
        mdl.minimize(mdl.sum(all_charging_slack_vars))
        # 求解模型
        solution = mdl.solve()

        # 打印解决方案
        if solution:
            # print("充电循环计算结束")
            # 分析每辆车的充放电活动
            for vehicle_id in range(self.ev_num):
                prev_state = - 1  # 前一个时刻的充放电状态，初始化为0（无活动）
                current_cycle_charge = - 0.6  # 当前充放电周期的累计SOC
                current_soc = 0.3 #当前SOC
                inflection_soc = [0.9] #拐点SOC
                cycle_sign = 0 #

                for t in range(48):
                    charging_state = solution.get_value(charging_vars[vehicle_id][t]) + solution.get_value(charging_slack_vars[vehicle_id][t])
                    discharging_state = solution.get_value(discharging_vars[vehicle_id][t]) + solution.get_value(discharging_slack_vars[vehicle_id][t])
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

                        inflection_soc = [current_soc]  # 重置当周期记录

                    if current_state != 0:
                        prev_state = current_state  # 更新前一个时刻的状态
                    else:
                        prev_state = prev_state

        else:
            # 获取求解器状态
            status = mdl.get_solve_status()
            print(f"{self.node_num}Solver status:", status)
            error_day.append([self.node_num,day])


        # # 快充
        # cycle_quick = EV_charging_quick
        # vehicle_cycles_info[0.6] += cycle_quick

        # # 转换为数据框
        # df_charge = pd.DataFrame(list(vehicle_cycles_info.items()), columns=['Parameter', 'Value'])
        # # 定义文件路径
        # aging_folder_path = os.path.join(aging_path, str(self.ev_p))
        # if not os.path.exists(aging_folder_path):
        #     # 如果不存在，创建文件夹
        #     os.makedirs(aging_folder_path)
        # file_name = os.path.join(aging_folder_path, f"{self.node_num}.csv")
        # # 保存数据框为 CSV 文件
        # df_charge.to_csv(file_name, index=False)
        # if error_day:
        #     # 将 error_day 转换为 DataFrame，并设置列名
        #     df_error_day = pd.DataFrame(error_day, columns=["node_num", "day"])
        #
        #     # 保存 DataFrame
        #     df_error_day.to_csv("error_day.csv", index=False)  # 示例：将其保存为 CSV 文件
        # # 保存到 CSV 文件
        # file_name_error = os.path.join(aging_folder_path, f"{self.node_num}_error.csv")
        # print("已保存文件:", file_name)


ev_p = 0.15
node_num = 304
day = 111
sim = Node_annual(node_num, ev_p)
sim.year_sim(day)
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

