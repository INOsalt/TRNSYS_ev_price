import os
from tqdm import tqdm
from gridinfo import nodes, initial_EV, node_mapping,start_points,end_points,C_buy,C_sell,prices_real,expand_array
# from docplex.mp.model import Model
# v2g order
from anode_EVloadDOC_v2gorder import EVload_node
# order
# from anode_EVloadDOC_order import EVload_node
# import importlib
import pandas as pd
import numpy as np
# import math


# # 定义EV_penetration和v2g_ratio的可能值
# EV_penetration_values = [800 * x for x in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]] #
# v2g_ratio_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9] #




# ev_p = 0.15
v2g = 0
daily_average = 24.4 #单程平均出行距离 12.2 公里
charging_delta = 0.11 #1/(70*0.6/0.188/24.4)
EV_buy_price = expand_array(np.array(prices_real))
EV_sell_price = max(prices_real)

aging_path = "vehicle_cycles/LOAD_v2g"
# 功率曲线储存
power_path = 'data_annual_LOAD_v2g/'
# EV信息储存
EV_path = 'annual_EV_LOAD_v2g/'
# 读取CSV文件
folder_path = 'annual_onlyLOAD/'
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
        datapower_dir = power_path + f'{self.ev_p}'
        if not os.path.exists(datapower_dir):
            # 如果不存在，创建文件夹
            os.makedirs(datapower_dir)

        dataEV_dir = EV_path + f'{self.ev_p}'
        if not os.path.exists(dataEV_dir):
            # 如果不存在，创建文件夹
            os.makedirs(dataEV_dir)
        EV_charge_cost = []
        EV_discharge_cost = []
        EV_pay = []
        EV_pay_average = []
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
                        self.ev_p,v2g,datapower_dir,nodedata_dict, re_capacity_dict,self.node_num,self.ev_num)

            a = sum(charge_values)
            b = sum(discharge_values)
            c = self.ev_num * 12

            charge_values = np.round(np.array(charge_values))
            discharge_values = np.round(np.array(discharge_values))
            cost_charge = np.sum(charge_values * EV_buy_price) * 0.5
            cost_discharge = np.sum(discharge_values * EV_sell_price) * 0.5
            EV_charge_cost.append(cost_charge)
            EV_discharge_cost.append(cost_discharge)
            owner_pay = cost_charge - cost_discharge
            EV_pay.append(owner_pay)
            owner_pay_average = owner_pay/self.ev_num
            EV_pay_average.append(owner_pay_average)

            # 存储计算结果到CSV文件
            # 追加node_P_total和node_P_basic_and_EV到对应的文件
            append_to_csv(node_P_total, f'{datapower_dir}/{self.node_num}_P_total.csv')
            append_to_csv(node_P_basic_and_EV, f'{datapower_dir}/{self.node_num}_P_basic_and_EV.csv')

            append_to_csv(charge_values, f'{dataEV_dir}/{self.node_num}_charge_values.csv')
            append_to_csv(discharge_values, f'{dataEV_dir}/{self.node_num}_discharge_values.csv')

        df_cost = pd.DataFrame({
            "EV_charge_cost": EV_charge_cost,
            "EV_discharge_cost": EV_discharge_cost,
            "EV_pay": EV_pay,
            "EV_pay_average": EV_pay_average
        })

        # Save to CSV
        df_cost.to_csv(f'{dataEV_dir}/EV_cost_data_{self.node_num}.csv', index=False)  # Save without row indices



# for ev_p in tqdm([0.5, 1], desc="Outer loop over ev_p"): #0.15, 0.3,
#     for node_num in tqdm(start_points, desc="Inner loop over nodes", leave=False):
#         sim = Node_annual(node_num, ev_p)
#         sim.year_sim()
ev_p = 0.5
for node_num in tqdm([311, 312, 313, 314, 315, 316, 317,
                      318, 401, 402, 403, 404, 405, 407], desc="Inner loop over nodes", leave=False):
    sim = Node_annual(node_num, ev_p)
    sim.year_sim()



# sim = Node_annual(202)
# sim.year_sim()
# 202, 203, 204, 205, 206, 207, 208, 209, 301, 302,
#                       303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
#                       313, 314, 315, 316, 317, 318, 401, 402, 403, 404,
#                       405, 407]

