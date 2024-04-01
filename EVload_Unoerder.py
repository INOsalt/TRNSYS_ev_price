from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
import numpy as np
from charging_choice import ChargingManager
from gridinfo import (end_points, nodes, node_mapping, reverse_node_mapping, transition_matrices, prices_real)
#from runner import EV_penetration, v2g_ratio
import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_charging_distribution(charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix, not_charging_matrix): # 48行
    # 初始化存储结果的字典
    charging_distribution_work_slow = {}
    charging_distribution_work_quick = {}
    charging_distribution_home = {}
    not_charging_distribution = {}

    # 获取除了end_points之外的所有节点的索引
    non_end_point_indices = [idx for node, idx in node_mapping.items() if node not in end_points]

    # 反向映射：从矩阵索引到节点编号
    # reverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # 处理charging_work_slow_matrix，为end_points对应的列提取数据
    for end_point in end_points:
        idx = node_mapping[end_point]
        charging_distribution_work_slow[end_point] = charging_work_slow_matrix[:, idx]

    # 处理charging_work_quick_matrix，为end_points对应的列提取数据
    for end_point in end_points:
        idx = node_mapping[end_point]
        charging_distribution_work_quick[end_point] = charging_work_quick_matrix[:, idx]

    # 处理charging_home_matrix，为除了end_points之外的节点提取数据
    for idx in non_end_point_indices:
        node = reverse_node_mapping[idx]
        charging_distribution_home[node] = charging_home_matrix[:, idx]

    # 处理 not_charging_matrix，为所有节点提取
    for node in nodes:
        idx = node_mapping[node]
        not_charging_distribution[node] = not_charging_matrix[:, idx]

    return charging_distribution_work_slow, charging_distribution_work_quick, charging_distribution_home, not_charging_distribution


def calculate_leaving_vehicles(charging_distribution):
    leaving_vehicles = {}

    for node, charging_vector in charging_distribution.items():
        node_idx = node_mapping[node]  # 获取节点在转移矩阵中的索引
        leaving_vector = np.zeros_like(charging_vector)
        for i, vehicles_at_time in enumerate(charging_vector):
            transition_matrix = transition_matrices[i]  # 获取对应的转移矩阵
            Pjj = transition_matrix[node_idx, node_idx]  # 获取节点自身的转移概率
            leaving_vector[i] = (1 - Pjj) * vehicles_at_time  # 计算离开的车辆数量
        leaving_vehicles[node] = leaving_vector

    return leaving_vehicles


def calculate_arriving_vehicles(charging_distribution, leaving_vehicles):
    # 四舍五入后的字典
    charging_distribution_round = {node: np.round(vector).astype(int) for node, vector in charging_distribution.items()}
    leaving_vehicles_round = {node: np.round(vector).astype(int) for node, vector in leaving_vehicles.items()}

    # 计算到达的车辆
    arriving_vehicles = {}
    for node, vector in charging_distribution_round.items():
        arriving_vector = np.zeros_like(vector)
        for i in range(len(vector) - 1):
            leaving_vector = leaving_vehicles_round[node]
            p_now = vector[i]
            p_next = vector[i + 1]
            leaving = leaving_vector[i]
            arriving_vector[i] = p_next - p_now + leaving
            #arriving_vector[i] = vector[i + 1] - vector[i] + leaving_vector[i]
        # 最后一个时间点的到达车辆设置为0
        arriving_vector[-1] = 0
        arriving_vehicles[node] = np.round(arriving_vector).astype(int)

    return charging_distribution_round, leaving_vehicles_round, arriving_vehicles

def calculate_P_basic(nodedata_dict, re_capacity_dict):
    P_basic_dict = {}
    for hour in range(24):
        load_matrix = nodedata_dict[hour]
        re_matrix = re_capacity_dict.get(hour, np.zeros_like(load_matrix))


        for node in nodes:
            node_index = np.where(load_matrix[:, 0] == node)[0][0]
            load = load_matrix[node_index, 1]

            # 对于pv和wt，检查节点是否在对应的矩阵中
            if node in re_matrix[:, 0]:
                pv_index = np.where(re_matrix[:, 0] == node)[0][0]
                re = re_matrix[pv_index, 1]
            else:
                re = 0

            net_load = load - re

            if node not in P_basic_dict:
                P_basic_dict[node] = [net_load] * 2
            else:
                P_basic_dict[node].extend([net_load] * 2)

    return P_basic_dict


def P_EV_no_order(work_slow_charging_distribution, home_charging_distribution,
                  work_quick_charging_distribution, work_slow_arriving, home_arriving):
    CAP_BAT_EV = 42  # 固定的每辆车充电需求（70kWh*0.6）
    DELTA_T = 0.5  # 每个时间段长度，30分钟
    N_SLOTS = 48  # 一天中的时间段数量
    P_slow = 7  # kW
    P_quick = 42  # kW
    efficiency = 0.9
    # 初始化存储每个节点包括EV负载后的总负载的字典
    node_P_basic_and_EV = {}
    node_EV_load = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时所有节点EV slow负荷的字典
    node_EV_slowload = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时所有节点快充EV负荷的字典
    node_quick_EV_load = {time: np.zeros(len(nodes)) for time in range(48)}


    for node in nodes:
        EV_load = [0] * N_SLOTS  # 初始化EV负载列表
        EV_slowload = [0] * N_SLOTS  # 初始化EV负载列表
        EV_quickload = [0] * N_SLOTS  # 初始化EV负载列表

        if 100 <= node < 199:  # Office节点
            slow_charging_slots = work_slow_charging_distribution.get(node, [0] * N_SLOTS)
            quick_charging_slots = work_quick_charging_distribution.get(node, [0] * N_SLOTS)
            arriving_slots = work_slow_arriving.get(node, [0] * (N_SLOTS-1))

            # 快充车辆充电计算
            for t in range(N_SLOTS):
                EV_load[t] += quick_charging_slots[t] * P_quick / efficiency
                EV_quickload[t] = quick_charging_slots[t] * P_quick

            # 0时刻的车辆充电处理
            for i in range(0, 12):
                if i < N_SLOTS:
                    EV_load[i] += slow_charging_slots[0] * P_slow / efficiency
                    EV_slowload[i] = slow_charging_slots[0] * P_slow

            # 从1时刻开始使用arriving_slots
            for t in range(1, N_SLOTS):
                if t-1 < len(arriving_slots):
                    arriving_cars = arriving_slots[t-1]
                    for i in range(t, min(t + 12, N_SLOTS)):
                        EV_load[i] += arriving_cars * P_slow / efficiency
                        EV_slowload[i] += arriving_cars * P_slow
                        if np.sum(EV_load[:i + 1]) >= sum(arriving_slots) * CAP_BAT_EV:
                            break
                if np.sum(EV_load) * efficiency >= sum(arriving_slots) * CAP_BAT_EV:
                    break

        else:  # Home节点
            slow_charging_slots = home_charging_distribution.get(node, [0] * N_SLOTS)
            arriving_slots = home_arriving.get(node, [0] * (N_SLOTS-1))

            # 0时刻的车辆充电处理
            for i in range(0, 12):
                if i < N_SLOTS:
                    EV_load[i] += slow_charging_slots[0] * P_slow / efficiency
                    EV_slowload[i] += slow_charging_slots[0] * P_slow

            # 从1时刻开始使用arriving_slots
            for t in range(1, N_SLOTS):
                if t-1 < len(arriving_slots):
                    arriving_cars = arriving_slots[t-1]
                    for i in range(t, min(t + 12, N_SLOTS)):
                        EV_load[i] += arriving_cars * P_slow / efficiency
                        EV_slowload[i] += arriving_cars * P_slow
                        #检查负载是否够了
                        if np.sum(EV_load[:i + 1]) >= sum(arriving_slots) * CAP_BAT_EV:
                            break
                if np.sum(EV_load) >= sum(arriving_slots) * CAP_BAT_EV:
                    break

        # 将EV负载与基础负载相加得到总负载
        P_EV = [EV_load[t] for t in range(N_SLOTS)]
        node_P_basic_and_EV[node] = P_EV

        node_idx = node_mapping[node]
        for t in range(48):
            node_EV_load[t][node_idx] = P_EV[t]
            node_quick_EV_load[t][node_idx] = EV_quickload[t]
            node_EV_slowload[t][node_idx] = EV_slowload[t]


    return node_EV_load, node_EV_slowload, node_quick_EV_load


# 定义一个辅助函数用于保存包含向量值的字典到CSV
def save_dict_to_csv(base_path, data_dict, filename):
    # 准备空的DataFrame
    df = pd.DataFrame()
    for key, vector in data_dict.items():
        # 将每个键值对转换为DataFrame，其中键作为一列，向量展开为多行
        temp_df = pd.DataFrame({f'Node_{key}': vector})
        # 将每个新DataFrame作为列加入到最终的DataFrame中
        df = pd.concat([df, temp_df], axis=1)

    # 保存到CSV
    df.to_csv(os.path.join(base_path, f'{filename}.csv'), index=False)

def EVload(EV_Q1, EV_S1, EV_2, EV_3, EV_4,EV_penetration,v2g_ratio,file_path, nodedata_dict, re_capacity_dict):
    # 初始化存储每个微电网48步长EV负荷的字典
    mic_EV_load_slow = {mic: np.zeros(48) for mic in range(4)}  # 4个微电网
    # 初始化存储每个微电网48步长EV负荷的字典
    mic_EV_load_quick = {mic: np.zeros(48) for mic in range(4)}  # 4个微电网

    # 充电选择实例
    EV_choice = ChargingManager(EV_Q1, EV_S1, EV_2, EV_3, EV_4,EV_penetration,v2g_ratio)
    # 充电矩阵
    charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix, not_charging_matrix \
        = EV_choice.calculate_vehicle_distribution()
    # 处理为字典
    charging_distribution_work_slow, charging_distribution_work_quick, charging_distribution_home, distribution_not_charging \
        = extract_charging_distribution(charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix, not_charging_matrix)
    #计算离开和到达
    # 工作慢充
    leaving_vehicles_work_slow = calculate_leaving_vehicles(charging_distribution_work_slow)
    work_slow_charging_distribution, work_slow_leaving, work_slow_arriving\
        = calculate_arriving_vehicles(charging_distribution_work_slow, leaving_vehicles_work_slow)
    # 家慢充
    leaving_vehicles_home = calculate_leaving_vehicles(charging_distribution_home)
    home_charging_distribution, home_leaving, home_arriving \
        = calculate_arriving_vehicles(charging_distribution_home, leaving_vehicles_home)
    # 不充电
    leaving_not_charging = calculate_leaving_vehicles(distribution_not_charging)
    not_charging_distribution, not_charging_leaving, not_charging_arriving \
        = calculate_arriving_vehicles(distribution_not_charging, leaving_not_charging)
    # 工作快充
    work_quick_charging_distribution = {node: np.round(vector).astype(int) for node, vector
                                        in charging_distribution_work_quick.items()}

    # 定义一个辅助函数用于保存字典到CSV
    base_path = os.path.join(file_path, str(EV_penetration), str(v2g_ratio))
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 使用辅助函数保存每个字典
    save_dict_to_csv(base_path, work_slow_charging_distribution, 'work_slow_charging_distribution')
    save_dict_to_csv(base_path,work_slow_leaving, 'work_slow_leaving')
    save_dict_to_csv(base_path,work_slow_arriving, 'work_slow_arriving')

    save_dict_to_csv(base_path,home_charging_distribution, 'home_charging_distribution')
    save_dict_to_csv(base_path,home_leaving, 'home_leaving')
    save_dict_to_csv(base_path,home_arriving, 'home_arriving')

    save_dict_to_csv(base_path,not_charging_distribution, 'not_charging_distribution')
    save_dict_to_csv(base_path,not_charging_leaving, 'not_charging_leaving')
    save_dict_to_csv(base_path,not_charging_arriving, 'not_charging_arriving')

    save_dict_to_csv(base_path,work_quick_charging_distribution, 'work_quick_charging_distribution')

    # Pbasic字典
    P_basic_dict = calculate_P_basic(nodedata_dict, re_capacity_dict)

    #没有优化的负载：
    node_P_EV, node_EV_slowload, node_quick_EV_load = P_EV_no_order(work_slow_charging_distribution, home_charging_distribution,
                              work_quick_charging_distribution, work_slow_arriving, home_arriving)

    # 对每个节点，按照节点范围汇总微电网的EV负荷
    for node in nodes:
        # 获取当前节点的索引
        node_idx = node_mapping[node]
        # 确定当前节点属于哪个微电网
        mic_idx = None
        if 100 <= node <= 199:
            mic_idx = 0
        elif 200 <= node <= 299:
            mic_idx = 1
        elif 300 <= node <= 399:
            mic_idx = 2
        elif 400 <= node <= 499:
            mic_idx = 3

        # 如果节点属于某个微电网，更新对应微电网的负荷
        if mic_idx is not None:
            for t in range(48):
                mic_EV_load_slow[mic_idx][t] += node_EV_slowload[t][node_idx]
                mic_EV_load_quick[mic_idx][t] += node_quick_EV_load[t][node_idx]


    print("EV计算结束")
    return node_P_EV, mic_EV_load_slow, mic_EV_load_quick, P_basic_dict

class EVLoadVisualization:
    def __init__(self, P_BASIC, P_total):
        self.P_BASIC = P_BASIC
        self.P_total = P_total

    def plot_load_curves(self, title='power network load peaking'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.P_BASIC, label='P_BASIC', linestyle='--', marker='o', color='blue')
        plt.plot(self.P_total, label='P_total', linestyle='-', marker='x', color='red')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Load (kW)')
        # 设置x轴的刻度标签，每两个标记一个
        xticks_positions = range(0, len(self.P_BASIC), 2)
        xticks_labels = [f'{t}' for t in xticks_positions]
        plt.xticks(xticks_positions, xticks_labels)
        plt.legend()
        plt.grid(True)
        plt.show()

# # #实例调用
# EV_Q1 = np.array(prices_real)
# EV_S1 = np.array(prices_real)
# EV_2 = np.array(prices_real)
# EV_3 = np.array(prices_real)
# EV_4 = np.array(prices_real)
#
# EV_p = 800 * 0.1
# v2g = 0.5
#
# node_P_EV, mic_EV_load_slow, mic_EV_load_quick, P_basic_dict = EVload(EV_Q1, EV_S1, EV_2, EV_3, EV_4,EV_p,v2g,'file_path')
# print(node_P_EV)

# # # 创建可视化类的实例
# # # Pbasic字典
# # node = 101
# # P_TOTAL = node_P_total[node]
# # P_BASIC = node_P_basic_and_EV[node]
# # P_SLACK = node_slack_load[node]
# # visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# # #绘制和查看曲线
# # visualizer.plot_load_curves()
# # print(sum(P_TOTAL))
# # print(sum(P_BASIC))
# # print(sum(P_SLACK))
#
# node = 102
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))
#
# node = 206
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))

# node = 205
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))
#
# node = 305
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))
# #
# node = 317
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))
#
# node = 318
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))
#
# node = 401
# P_TOTAL = node_P_total[node]
# P_BASIC = node_P_basic_and_EV[node]
# P_SLACK = node_slack_load[node]
# visualizer = EVLoadVisualization(P_BASIC=P_BASIC, P_total=P_TOTAL)
# #绘制和查看曲线
# visualizer.plot_load_curves()
# print(sum(P_TOTAL))
# print(sum(P_BASIC))
# print(sum(P_SLACK))