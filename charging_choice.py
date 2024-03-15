import numpy as np
import itertools
# import matplotlib.pyplot as plt
from gridinfo import (C_buy, C_sell, start_points, end_points, microgrid_id, nodes, node_mapping,
                      transition_matrices, charge_ratio)

class CommuterChargingChoiceCalculator:
    def __init__(self, home_prices, work_prices, work_slowprices, work_slowprices1, work_slowprices2, EV_penetration):
        self.home_prices = home_prices
        self.work_prices = work_prices
        self.num_owners = EV_penetration * charge_ratio #ChargingManager 有不需要充电的车，需要互补
        self.work_slowprices1 = work_slowprices1
        self.work_slowprices2 = work_slowprices2
        self.work_slowprices = work_slowprices
        #self.home_charging_capacity = 120
        #self.work_charging_capacity = 200

    def calculate_home_charging_cost(self):
        # 选择社区价格数组0-8和20-23中最高的6个数相加
        night_prices = np.concatenate((self.home_prices[0:8], self.home_prices[20:24]))
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_private_charging_cost(self):
        # 选择社区价格数组0-8和20-23中最高的6个数相加
        night_prices = np.concatenate((C_buy[0:8], C_buy[20:24]))
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_workquick_charging_cost(self):
        # 从公司价格数组9-19点中找出最小的一个价格
        daytime_prices = self.work_prices[9:19]
        min_sum = np.min(daytime_prices)
        return min_sum
    def calculate_workslow_charging_cost(self):
        # 选择社区价格数组9-19中最高的6个数相加
        day_prices = np.sort(self.work_slowprices[9:19])[-6:]
        highest_prices = np.sum(day_prices)
        return np.sum(highest_prices)
    def calculate_workslow_charging_cost1(self):
        # 选择社区价格数组9-19中最高的6个数相加
        day_prices = np.sort(self.work_slowprices[9:19])[-6:]
        highest_prices = np.sum(day_prices)
        return np.sum(highest_prices)

    def calculate_workslow_charging_cost2(self):
        # 选择社区价格数组9-19中最高的6个数相加
        day_prices = np.sort(self.work_slowprices[9:19])[-6:]
        highest_prices = np.sum(day_prices)
        return np.sum(highest_prices)

    def calculate_choices(self):
        # 计算各种充电方式的成本
        home_charging_cost = self.calculate_home_charging_cost()
        work_quick_charging_cost = self.calculate_workquick_charging_cost()
        work_slow_charging_cost = self.calculate_workslow_charging_cost()
        work_slow_charging_cost1 = self.calculate_workslow_charging_cost1()
        work_slow_charging_cost2 = self.calculate_workslow_charging_cost2()
        private_charging_cost = self.calculate_private_charging_cost()

        # 初始化各充电方式的车主数量
        choices = {
            'charging_at_home_private': self.num_owners // 2,  # 一半车主使用私人充电桩
            'charging_at_home_public': 0,
            'charging_at_work_quick': 0,
            'charging_at_work_slow': 0,
            'charging_at_work_slow1': 0,
            'charging_at_work_slow2': 0,
        }

        # 另一半车主决定充电地点
        decision_making_owners = self.num_owners - choices['charging_at_home_private']

        # 根据不同群体计算充电选择
        for owner_group, fraction in [('group1', 1 / 3), ('group2', 1 / 6), ('group3', 1 / 6), ('group4', 1 / 3)]:
            num_owners_in_group = decision_making_owners * fraction

            if owner_group == 'group1':
                # 四种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost1]
            elif owner_group == 'group2':
                # 四种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost2]
            elif owner_group == 'group3':
                # 五种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost1, work_slow_charging_cost2]
            else:  # group4
                # 三种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost]

            # 选择成本最低的充电方式
            min_cost_option = options.index(min(options))

            # 根据最低成本选项分配车主
            if min_cost_option == 0:
                choices['charging_at_home_public'] += num_owners_in_group
            elif min_cost_option == 1:
                choices['charging_at_work_quick'] += num_owners_in_group
            elif min_cost_option == 2:
                choices['charging_at_work_slow'] += num_owners_in_group
            elif min_cost_option == 3 and 'work_slow_charging_cost1' in locals():
                choices['charging_at_work_slow1'] += num_owners_in_group
            elif min_cost_option == 4 and 'work_slow_charging_cost2' in locals():
                choices['charging_at_work_slow2'] += num_owners_in_group

        return choices


class ChargingManager:
    def __init__(self, MGQ1, MGS1, MG2, MG3, MG4, EV_penetration, v2g_ratio):#元/kWh
        self.P_slow = 7  # kW
        self.P_quick = 42  # kW
        self.DELTA_T = 0.5  # 每个时间段长度，30分钟
        self.num_owners_not_charging = EV_penetration * v2g_ratio
        self.EV_penetration = EV_penetration
        self.MGQ1 = MGQ1 * self.P_quick * self.DELTA_T # 每个步长价格
        self.MGS1 = MGS1 * self.P_slow * self.DELTA_T
        self.MG2 = MG2 * self.P_slow * self.DELTA_T
        self.MG3 = MG3 * self.P_slow * self.DELTA_T
        self.MG4 = MG4 * self.P_slow * self.DELTA_T
        # 创建节点到索引的映射
        self.node_mapping = node_mapping
        # 初始化向量
        self.charging_home = np.zeros(len(nodes)*2)  # 80
        self.charging_work_slow = np.zeros(len(nodes)*2)  # 80
        self.charging_work_quick = np.zeros(len(nodes))  # 40
        # 不充电车
        self.not_charging = np.zeros(len(nodes) * 2)

        # 初始化特殊点充电车辆数记录字典
        self.special_slow_charging_counts = {201: 0, 207: 0, 205: 0, 301: 0, 311: 0, 312: 0}

    def calculate_charging_distribution(self):
        # 遍历起点，计算充电选择，并更新向量
        for start_point in start_points:
            microgrid = microgrid_id(start_point)
            home_price = self.MG2 if microgrid == 1 else self.MG3 if microgrid == 2 else self.MG4

            # 使用CommuterChargingChoiceCalculator计算选择
            calculator = CommuterChargingChoiceCalculator(home_price, self.MGQ1, self.MGS1, self.MG2, self.MG3, self.EV_penetration)
            choices = calculator.calculate_choices()

            # 根据映射更新向量
            idx = self.node_mapping[start_point]
            # 不充电车
            self.not_charging[idx] = self.num_owners_not_charging
            self.charging_home[idx] = choices['charging_at_home_private'] + choices['charging_at_home_public']
            # 更新慢充向量
            self.charging_work_slow[idx] += (choices['charging_at_work_slow'] +
                                             choices['charging_at_work_slow1'] +
                                             choices['charging_at_work_slow2'])

            # 特殊处理慢充1和慢充2
            special_slow1 = choices['charging_at_work_slow1']
            special_slow2 = choices['charging_at_work_slow2']

            if special_slow1 > 0 and special_slow2 > 0:
                if special_slow1 > special_slow2:
                    # special_slow1 大于 special_slow2 的情况
                    for point in [201, 207, 205]:
                        self.special_slow_charging_counts[point] += special_slow1 / 3
                    self.special_slow_charging_counts[312] += special_slow2
                    self.special_slow_charging_counts[301] += 0
                    self.special_slow_charging_counts[311] += 0
                    # 301, 311 是 0，不需要额外处理
                else:
                    # special_slow2 大于或等于 special_slow1 的情况
                    for point in [201, 207]:
                        self.special_slow_charging_counts[point] += special_slow1 / 2
                    # 205 是 0，不需要额外处理
                    self.special_slow_charging_counts[301] += special_slow2 / 4
                    self.special_slow_charging_counts[311] += special_slow2 / 4
                    self.special_slow_charging_counts[312] += special_slow2 / 2
                    #self.charging_work_slow[self.node_mapping[312]] += special_slow2 / 2
            else:
                # 保持原始规则不变
                for point in [201, 207, 205]:
                    self.special_slow_charging_counts[point] += special_slow1 / 3
                    self.charging_work_slow[self.node_mapping[point]] += special_slow1 / 3
                self.special_slow_charging_counts[301] += special_slow2 / 4
                self.special_slow_charging_counts[311] += special_slow2 / 4
                self.special_slow_charging_counts[312] += special_slow2 / 2

            # 给end_points的每个点的'charging_at_work_quick'均分加上
            quick_share = choices['charging_at_work_quick'] / len(end_points)
            for end_point in end_points:
                self.charging_work_quick[self.node_mapping[end_point]] += quick_share

    def calculate_vehicle_distribution(self):
        # 初始化结果矩阵
        self.charging_home_matrix = np.zeros((48, 80))
        self.charging_work_slow_matrix = np.zeros((48, 80))
        self.not_charging_matrix = np.zeros((48, 80))
        self.charging_work_quick_matrix = np.zeros((48, 40))
        # 调用计算车主选择
        self.calculate_charging_distribution()

        # 首先，初始化每个状态矩阵的第一行为对应的初始状态
        self.charging_home_matrix[0, :] = self.charging_home
        self.charging_work_slow_matrix[0, :] = self.charging_work_slow
        self.not_charging_matrix[0, :] = self.not_charging

        # 遍历每个转移矩阵，从第二行开始更新状态矩阵
        # 迭代变量从`t`改为`i`和`matrix`，并且从1开始迭代，因为第0行已经设置
        for i, (t, matrix) in enumerate(itertools.islice(transition_matrices.items(), 0, 47), start=1):
            self.charging_home_matrix[i, :] = np.dot(self.charging_home_matrix[i - 1, :], matrix)
            self.charging_work_slow_matrix[i, :] = np.dot(self.charging_work_slow_matrix[i - 1, :], matrix)
            self.not_charging_matrix[i, :] = np.dot(self.not_charging_matrix[i - 1, :], matrix)

        # 截取前40列
        self.charging_home_matrix = self.charging_home_matrix[:, :40]
        self.charging_work_slow_matrix = self.charging_work_slow_matrix[:, :40]
        self.not_charging_matrix = self.not_charging_matrix[:, :40]

        # 根据special_slow_charging_counts修正self.charging_work_slow_matrix和self.charging_home_matrix
        # 计算特殊慢充的比例
        special_slow_charging_ratios = {}
        for node, count in self.special_slow_charging_counts.items():
            special_slow_charging_ratio = count / (self.EV_penetration * charge_ratio * len(start_points) / 6)
            special_slow_charging_ratios[node] = special_slow_charging_ratio

        # 修正self.charging_work_slow_matrix 和 self.charging_home_matrix
        # 定义需要调整的行索引，对应9-19点
        rows_to_adjust1 = list(range(18, 40))  # 9-19点对应的索引，包括9点开始到19点结束的所有半小时段

        adjustment_nodes = {
            201: 102,
            207: 103,
            312: 106,
            205: 104,
            301: 104,
            311: 104
        }

        for adjust_node, source_node in adjustment_nodes.items():
            source_idx = self.node_mapping[source_node]
            adjust_idx = self.node_mapping[adjust_node]

            # 计算减少的部分，只针对9-19点的时间段
            reduced_amount = self.charging_work_slow_matrix[rows_to_adjust1, source_idx] * special_slow_charging_ratios[
                adjust_node]

            # 减少工作慢充中的相应部分，只针对9-19点的时间段
            self.charging_work_slow_matrix[rows_to_adjust1, source_idx] -= reduced_amount

            # 将减少的部分加到家充对应的adjust_node，只针对9-19点的时间段
            self.charging_home_matrix[rows_to_adjust1, adjust_idx] += reduced_amount

        # 修正 self.charging_home_matrix # adjust - source
        adjustment_nodes2to3 = {310: 204}
        adjustment_nodes4to3 = {
            313: 401,
            315: 402,
            316: 403,
            318: 404,
        }
        # 234区价格
        calculator_MG2 = CommuterChargingChoiceCalculator(self.MG2, self.MGQ1, self.MGS1, self.MG2, self.MG3, self.EV_penetration)
        calculator_MG3 = CommuterChargingChoiceCalculator(self.MG3, self.MGQ1, self.MGS1, self.MG2, self.MG3, self.EV_penetration)
        calculator_MG4 = CommuterChargingChoiceCalculator(self.MG4, self.MGQ1, self.MGS1, self.MG2, self.MG3, self.EV_penetration)
        PRICE_MG2 = calculator_MG2.calculate_home_charging_cost()
        PRICE_MG3 = calculator_MG3.calculate_home_charging_cost()
        PRICE_MG4 = calculator_MG4.calculate_home_charging_cost()

        # 定义需要调整的行索引，对应0-8点和20-23点
        rows_to_adjust_am = list(range(0, 17))  # 0-8点，每小时分为两个半小时段，包括0点到8点的前一个半小时
        rows_to_adjust_pm = list(range(40, 48))  # 20-23点，每小时分为两个半小时段，包括20点到23点的最后一个半小时
        rows_to_adjust = rows_to_adjust_am + rows_to_adjust_pm

        if PRICE_MG2 > PRICE_MG3:  # 2减少3增加
            for tar, sour in adjustment_nodes2to3.items():  # target 3
                sour_idx = self.node_mapping[sour]
                tar_idx = self.node_mapping[tar]
                # 进行调整
                self.charging_home_matrix[rows_to_adjust, tar_idx] += self.charging_home_matrix[
                                                                          rows_to_adjust, sour_idx] / 2
                self.charging_home_matrix[rows_to_adjust, sour_idx] /= 2
        elif PRICE_MG2 < PRICE_MG3:
            # 相反的情况，把310对应的列中的0-8点，20-23行的一半挪到204对应的列
            for tar, sour in adjustment_nodes2to3.items():  # target 2
                sour_idx = self.node_mapping[tar]  # 这里应该是 source 的映射
                tar_idx = self.node_mapping[sour]  # 这里应该是 target 的映射，且需要反转source和target的角色
                # 进行调整
                self.charging_home_matrix[rows_to_adjust, tar_idx] += self.charging_home_matrix[
                                                                          rows_to_adjust, sour_idx] / 2
                self.charging_home_matrix[rows_to_adjust, sour_idx] /= 2
        # 如果PRICE_MG2 == PRICE_MG3，则不做任何调整

        # 对于MG4和MG3的价格比较
        if PRICE_MG4 > PRICE_MG3:
            for tar, sour in adjustment_nodes4to3.items():
                sour_idx = self.node_mapping[sour]
                tar_idx = self.node_mapping[tar]
                # 进行调整
                self.charging_home_matrix[rows_to_adjust, tar_idx] += self.charging_home_matrix[
                                                                          rows_to_adjust, sour_idx] / 2
                self.charging_home_matrix[rows_to_adjust, sour_idx] /= 2
        elif PRICE_MG4 < PRICE_MG3:
            # 如果MG4的价格小于MG3，需要调整的逻辑
            for tar, sour in adjustment_nodes4to3.items():  # 应使用4到3的映射关系
                sour_idx = self.node_mapping[sour]  # 正确设置source索引
                tar_idx = self.node_mapping[tar]  # 正确设置target索引
                # 进行调整
                self.charging_home_matrix[rows_to_adjust, tar_idx] += self.charging_home_matrix[
                                                                          rows_to_adjust, sour_idx] / 2
                self.charging_home_matrix[rows_to_adjust, sour_idx] /= 2
        # 如果PRICE_MG4 == PRICE_MG3，则不做任何调整

        # 快充矩阵：每辆车只停留1h

        # 找到MGQ1中9-19点间最小的值和与最小值相差小于等于0.1的时间点
        min_price = np.min(self.MGQ1[9:20])
        selected_hours = [i for i, price in enumerate(self.MGQ1[9:20], start=9) if price <= min_price + 0.1]
        # 转换选定的小时到半小时步长的索引，并且为每个小时选取两个半小时索引
        selected_indices = []
        for hour in selected_hours:
            selected_indices.extend([hour * 2, hour * 2 + 1])

        # 对于每个end_points对应的列，将end_points对应的快充数量均分到选定的时间点
        for end_point in end_points:
            idx = self.node_mapping[end_point]
            num_quick_charging = self.charging_work_quick[idx]  # 快充数量
            num_slots = len(selected_hours)

            # 分配快充数量到选定时间点
            for time_idx in selected_indices:
                self.charging_work_quick_matrix[time_idx, idx] = num_quick_charging / num_slots

        # 输出三个矩阵
        return self.charging_home_matrix, self.charging_work_slow_matrix, self.charging_work_quick_matrix, self.not_charging_matrix


# # 实例调用
# # 设置随机种子，以确保每次生成的随机数相同
# np.random.seed(42)
#
# # 生成5个长度为24的随机数组，并将它们赋值给对应的变量
# EV_Q1 = np.random.uniform(0.2205, 2, (24,))
# EV_S1 = np.random.uniform(0.2205, 2, (24,))
# EV_2 = np.random.uniform(0.2205, 2, (24,))
# EV_3 = np.random.uniform(0.2205, 2, (24,))
# EV_4 = np.random.uniform(0.2205, 2, (24,))
#
# a = ChargingManager(EV_Q1, EV_S1, EV_2, EV_3, EV_4)
# home, works, workq = a.calculate_vehicle_distribution()
# print(home)




# # EVDataVisualizer类: 可视化EV数据和充电模式
# class EVDataVisualizer:
#     def __init__(self):
#         pass
#
#     def printEVData(self, EV, title):
#         # EV数据可视化
#         plt.figure(figsize=(15, 5))
#
#         # EV到达时隙的频数直方图
#         plt.subplot(1, 3, 1)
#         plt.hist(EV['J_c'], bins=96, range=(0.5, 96.5))
#         plt.title('EV Arrival Time Slot Histogram - ' + title)
#         plt.xlabel('Arrival Time Slot')
#         plt.ylabel('Frequency')
#
#         # EV到达时间的频率直方图
#         plt.subplot(1, 3, 2)
#         plt.hist(EV['t_c'], bins=24, range=(0, 24), density=True)
#         plt.title('EV Arrival Time Frequency Histogram - ' + title)
#         plt.xlabel('Arrival Time (Hours)')
#         plt.ylabel('Frequency')
#
#         plt.tight_layout()
#         plt.show()
#
#         # EV电池状态SOC的散点图
#         plt.figure(figsize=(8, 5))
#         plt.scatter(np.arange(len(EV)), EV['SOC_con'], label='SOC Current', s=10)
#         plt.scatter(np.arange(len(EV)), EV['SOC_min'], label='SOC Minimum', s=10)
#         plt.scatter(np.arange(len(EV)), EV['SOC_max'], label='SOC Maximum', s=10)
#         plt.title('EV Battery State of Charge (SOC) Scatter Plot - ' + title)
#         plt.xlabel('EV Index')
#         plt.ylabel('State of Charge (SOC)')
#         plt.legend()
#         plt.show()
#
#     def figureResult(self, P_basic, P_SOC_crd, title):
#         # 充电模式结果可视化
#         plt.figure(figsize=(12, 6))
#         plt.plot(P_basic, label='Basic load')
#         plt.plot(P_SOC_crd, label='Optimized load', linestyle='--')
#         plt.xlabel('Time')
#         plt.ylabel('Load (KW)')
#         plt.title('Charging mode - ' + title)
#         plt.legend()
#         plt.show()
