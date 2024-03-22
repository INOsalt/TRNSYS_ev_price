from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
from tqdm import tqdm
import numpy as np
from charging_choice import ChargingManager
from gridinfo import (end_points, nodes, node_mapping, reverse_node_mapping, transition_matrices, prices_real)
import matplotlib.pyplot as plt
import pandas as pd
import os

class EVChargingOptimizer:
    def __init__(self):
        self.CAP_BAT_EV = 42  # 固定的每辆车充电需求（70kWh*0.6,70来自平均统计数据）
        self.DELTA_T = 0.5  # 每个时间段长度，30分钟
        self.N_SLOTS = 48  # 一天中的时间段数量
        self.P_slow = 7 # kW
        self.P_quick = 42 # kW
        self.efficiency = 1

    def optimizeCommunityChargingPattern(self, community_vehicles_distribution, community_arriving_vehicles,
                                         community_leaving_vehicles, community_P_BASIC, not_charging_distribution,
                                         leaving_not_charging):

        model = Model("EVChargingOptimization")

        # 决策变量
        MAX_EV = max(community_vehicles_distribution) # 最多车
        charge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="charge")
        discharge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="discharge")
        # 引入非负松弛变量,没有完全满足的对每个时段t
        slack_EV = {t: model.continuous_var(name=f"slack_{t}", lb=0) for t in range(self.N_SLOTS)}
        # v2g 变量
        MAX_EV_v2g = max(not_charging_distribution)
        v2g_charge1 = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV_v2g, name="v2g_charge")
        v2g_discharge1 = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV_v2g, name="v2g_discharge")
        # 为每个时段t创建一个二进制辅助变量
        z_home = model.binary_var_list(self.N_SLOTS, name="z")

        # 约束条件
        constraints = []

        # 约束0 v2g不能同时充放电
        for t in range(self.N_SLOTS):
            # state_charge[t] 可以大于0当且仅当 z[t] == 1
            model.add_constraint(v2g_charge1[t] <= MAX_EV * z_home[t])
            # state_discharge[t] 可以大于0当且仅当 z[t] == 0
            model.add_constraint(v2g_discharge1[t] <= MAX_EV * (1 - z_home[t]))

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        for t in range(self.N_SLOTS):
            constraints.append(model.sum(charge[t] + discharge[t]) <= community_vehicles_distribution[t])
            # 充电或放电的v2g车辆数不超过该时段车辆数
            constraints.append(model.sum(v2g_charge1[t] + v2g_discharge1[t])
                               <= not_charging_distribution[t])

        # 约束2：紧急需求车辆的充电要求
        for t in range(self.N_SLOTS):
            emergency_charging_needed = community_leaving_vehicles[t] * self.CAP_BAT_EV
            net_charging_provided = model.sum(
                (charge[t_prime] - discharge[t_prime]) * self.P_slow * self.DELTA_T
                for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(net_charging_provided + slack_EV[t] >= emergency_charging_needed)
            constraints.append(slack_EV[t] <= emergency_charging_needed * 0.1)
            # v2g 不能超过
            still_parking_EV = not_charging_distribution[t] - leaving_not_charging[t]
            net_charging = model.sum((v2g_charge1[t_prime] - v2g_discharge1[t_prime]) * self.P_slow * self.DELTA_T
                                     for t_prime in range(t + 1))  # 不平衡的车
            constraints.append(net_charging <= still_parking_EV * self.CAP_BAT_EV * 0.1)  # 如果是正数，最多充0.1 SOC
            constraints.append(net_charging >= - still_parking_EV * self.CAP_BAT_EV * 0.3)  # 如果是负数，最多放0.3 SOC

        # 约束3：总充电量需满足所有车辆的累计需求
        total_charging_demand = sum(community_leaving_vehicles) * self.CAP_BAT_EV
        # total_charging_demand_upper = sum(community_arriving_vehicles) * self.CAP_BAT_EV / 0.6 * 0.7
        # total_charging_demand_lower = sum(community_arriving_vehicles) * self.CAP_BAT_EV / 0.6 * 0.4
        total_charging_provided = model.sum(charge[t] * self.P_slow * self.DELTA_T
                                            for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(discharge[t] * self.P_slow * self.DELTA_T
                                              for t in range(self.N_SLOTS))
        total_slcak = model.sum(slack_EV[t] for t in range(self.N_SLOTS))# 没满足的功率
        # 确保总充电量（考虑放电减少的量）满足总需求
        constraints.append(total_charging_provided - total_discharging_reduced + total_slcak == total_charging_demand)
        # v2g 充电-放电 = 0
        total_charging_provided = model.sum(v2g_charge1[t] for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(v2g_discharge1[t] for t in range(self.N_SLOTS))
        constraints.append(total_charging_provided - total_discharging_reduced == 0)

        # 约束4：累计放电量不超过之前累计的充电量 累计净充电量不能超过
        for t in range(0, self.N_SLOTS):
            cumulative_charge_until_t = model.sum(charge[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t+1))
            cumulative_discharge_until_t = model.sum(discharge[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t+1))
            constraints.append(cumulative_discharge_until_t <= cumulative_charge_until_t)
            cumulative_leaving_t = np.sum(community_leaving_vehicles[0:t])
            constraints.append(cumulative_charge_until_t - cumulative_discharge_until_t
                               <= (community_vehicles_distribution[t] + cumulative_leaving_t) * self.CAP_BAT_EV)

        # v2g累计充电量不超过之前累计的放电量
            cumulative_charge_until_t = model.sum(v2g_charge1[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t + 1))  # 包括当前时段
            cumulative_discharge_until_t = model.sum(v2g_discharge1[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(cumulative_discharge_until_t >= cumulative_charge_until_t)

        model.add_constraints(constraints)

        # 计算每个时段的电网总负载，包括基本负载、充电增加和放电减少
        P_total = [community_P_BASIC[t] +
                   ((charge[t] + v2g_charge1[t]) * self.P_slow * self.DELTA_T / self.efficiency) -
                   ((discharge[t] + v2g_discharge1[t]) * self.P_slow * self.DELTA_T * self.efficiency)
                   for t in range(self.N_SLOTS)]

        # 在目标函数中加入惩罚项，惩罚未满足的紧急充电需求
        penalty_weight = 1000  # slack惩罚权重，尽量为0
        penalty_v2g = 0.01  # v2g惩罚权重，尽量小
        penalty_term = (model.sum(slack_EV[t] * penalty_weight for t in range(self.N_SLOTS))
                        + model.sum(v2g_charge1[t] * penalty_v2g for t in range(self.N_SLOTS))
                        + model.sum(v2g_discharge1[t] * penalty_v2g for t in range(self.N_SLOTS)))

        # 目标函数：最小化电网负载的高峰与低谷之间的差异
        model.minimize(model.max(P_total) - model.min(P_total) + penalty_term)
        #限制求解时间
        model.parameters.timelimit.set(120)

        # refiner = ConflictRefiner()  # 先实例化ConflictRefiner类
        # res = refiner.refine_conflict(model)  # 将模型导入该类，调用方法
        # res.display()
        solution = model.solve()

        if solution:
            # 初始化列表来存储每半小时的净功率和计算后的P_total
            ev_vector = [] #价格
            net_power_per_half_hour = [] #opf
            P_total_calculated = []  # 用于存储计算后的P_total
            total_slcak_list = []
            v2g_charge_values = []  # 存储v2g_charge的值
            v2g_discharge_values = []  # 存储v2g_discharge的值

            for t in range(self.N_SLOTS):
                # v2g 循环次数
                v2g_charge_values.append(solution.get_value(v2g_charge1[t]))
                v2g_discharge_values.append(solution.get_value(v2g_discharge1[t]))

                # 计算每个时段的充电功率总和，功率不用乘步长
                charging_power = solution.get_value(charge[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                discharging_power = solution.get_value(discharge[t]) * self.P_slow * self.efficiency
                # 计算v2g充电功率总和，功率不用乘步长
                v2g_charging_power = solution.get_value(v2g_charge1[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                v2g_discharging_power = solution.get_value(v2g_discharge1[t]) * self.P_slow * self.efficiency
                # 计算slack总和
                slack_power = solution.get_value(slack_EV[t]) / self.efficiency
                # 计算EV净功率：充电功率 - 放电功率 没有V2G
                ev_power = charging_power - discharging_power
                net_power = charging_power - discharging_power + v2g_charging_power- v2g_discharging_power
                net_power_per_half_hour.append(net_power)
                ev_vector.append(ev_power)

                # 同时计算这个时间段的P_total
                P_total_this_slot = community_P_BASIC[t] + charging_power - discharging_power
                P_total_calculated.append(P_total_this_slot)
                total_slcak_list.append(slack_power)

            # 返回计算好的net_power_per_half_hour和P_total_calculated
            return ev_vector, net_power_per_half_hour, P_total_calculated, total_slcak_list, v2g_charge_values, v2g_discharge_values
        else:
            # 如果没有找到解决方案，输出求解状态
            solve_status = model.get_solve_status()
            print(f"Solve status: {solve_status}")
            print("Community EVLOAD no solution found.")
            return {}

    def optimizeOfficeChargingPattern(self, slow_vehicles_distribution, slow_arriving_vehicles, slow_leaving_vehicles,
                                      fast_vehicles_distribution, office_P_BASIC, not_charging_distribution,
                                      leaving_not_charging):
        model = Model("OfficeEVChargingOptimization")

        # 决策变量
        # 慢充车辆充电状态
        MAX_EV = max(slow_vehicles_distribution)
        slow_charge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="charge")
        slow_discharge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="discharge")
        # 引入非负松弛变量,没有完全满足的对每个时段t
        slack_EV_work = {t: model.continuous_var(name=f"slack_{t}", lb=0) for t in range(self.N_SLOTS)}
        # v2g 变量
        MAX_EV_v2g = max(not_charging_distribution)
        v2g_charge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV_v2g, name="v2g_charge")
        v2g_discharge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV_v2g, name="v2g_discharge")
        # 为每个时段t创建一个二进制辅助变量
        z_work = model.binary_var_list(self.N_SLOTS, name="z_work")

        # 约束条件
        constraints = []

        # 约束0 v2g不能同时充放电
        for t in range(self.N_SLOTS):
            # state_charge[t] 可以大于0当且仅当 z[t] == 1
            model.add_constraint(v2g_charge[t] <= MAX_EV * z_work[t])
            # state_discharge[t] 可以大于0当且仅当 z[t] == 0
            model.add_constraint(v2g_discharge[t] <= MAX_EV * (1 - z_work[t]))

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            # 充电或放电的慢充车辆数不超过该时段车辆数
            constraints.append(model.sum(slow_charge[t] + slow_discharge[t])
                               <= slow_vehicles_distribution[t])
            # 充电或放电的v2g车辆数不超过该时段车辆数
            constraints.append(model.sum(v2g_charge[t] + v2g_discharge[t])
                               <= not_charging_distribution[t])

        # 约束2：紧急需求车辆的充电要求
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            emergency_charging_needed = slow_leaving_vehicles[t] * self.CAP_BAT_EV
            net_charging_provided = model.sum(
                (slow_charge[t_prime] - slow_discharge[t_prime]) * self.P_slow * self.DELTA_T
                for t_prime in range(t + 1))
            constraints.append(net_charging_provided + slack_EV_work[t] >= emergency_charging_needed)
            constraints.append(slack_EV_work[t] <= emergency_charging_needed * 0.1)
            # v2g 不能超过
            still_parking_EV = not_charging_distribution[t] - leaving_not_charging[t]
            net_charging = model.sum((v2g_charge[t_prime] - v2g_discharge[t_prime]) * self.P_slow * self.DELTA_T
                                     for t_prime in range(t + 1))  # 不平衡的车
            constraints.append(net_charging <= still_parking_EV * self.CAP_BAT_EV * 0.1)  # 如果是正数，最多充0.1 SOC
            constraints.append(net_charging >= - still_parking_EV * self.CAP_BAT_EV * 0.3)  # 如果是负数，最多放0.3 SOC

        # 约束3：总充电量需满足所有车辆的累计需求
        total_charging_demand = sum(slow_leaving_vehicles) * self.CAP_BAT_EV
        total_charging_provided = model.sum(slow_charge[t] * self.P_slow * self.DELTA_T
                                            for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(slow_discharge[t] * self.P_slow * self.DELTA_T
                                              for t in range(self.N_SLOTS))
        total_slcak = model.sum(slack_EV_work[t] for t in range(self.N_SLOTS)) # 没满足的功率
        # 确保总充电量（考虑放电减少的量）满足总需求
        constraints.append(total_charging_provided - total_discharging_reduced + total_slcak == total_charging_demand)
        # v2g 充电-放电 = 0
        total_charging_provided = model.sum(v2g_charge[t] for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(v2g_discharge[t] for t in range(self.N_SLOTS))
        constraints.append(total_charging_provided - total_discharging_reduced == 0)

        # 约束4：累计放电量不超过之前累计的净充电量
        for t in range(self.N_SLOTS):
            cumulative_charge_until_t = model.sum(slow_charge[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t + 1))  # 包括当前时段
            cumulative_discharge_until_t = model.sum(slow_discharge[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(cumulative_discharge_until_t <= cumulative_charge_until_t)
            # 充电量限制
            cumulative_leaving_t = np.sum(slow_leaving_vehicles[0:t])
            constraints.append(cumulative_charge_until_t - cumulative_discharge_until_t
                               <= (slow_vehicles_distribution[t] + cumulative_leaving_t) * self.CAP_BAT_EV)
            # v2g累计充电量不超过之前累计的放电量
            cumulative_charge_until_t = model.sum(v2g_charge[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t + 1))  # 包括当前时段
            cumulative_discharge_until_t = model.sum(v2g_discharge[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(cumulative_discharge_until_t >= cumulative_charge_until_t)

        model.add_constraints(constraints)

        # 目标函数
        # 在目标函数中加入惩罚项，惩罚未满足的紧急充电需求
        penalty_weight = 1000  # 惩罚权重，根据问题规模和重要性调整
        penalty_v2g = 0.01  # v2g惩罚权重，尽量小
        penalty_term = (model.sum(slack_EV_work[t] * penalty_weight for t in range(self.N_SLOTS))
                        + model.sum(v2g_charge[t] * penalty_v2g for t in range(self.N_SLOTS))
                        + model.sum(v2g_discharge[t] * penalty_v2g for t in range(self.N_SLOTS)))

        # 计算每个时段的电网总负载，考虑慢充充电、慢充放电和快充充电
        P_total = [office_P_BASIC[t] +
                   ((slow_charge[t] + v2g_charge) * self.P_slow * self.DELTA_T / self.efficiency
                    - (slow_discharge[t] + v2g_discharge) * self.P_slow * self.DELTA_T * self.efficiency) +
                   fast_vehicles_distribution[t] * self.P_quick * self.DELTA_T / self.efficiency
                   for t in range(self.N_SLOTS)]

        model.minimize(model.max(P_total) - model.min(P_total) + penalty_term)
        # 设置求解时间限制为120秒
        model.parameters.timelimit.set(120)

        solution = model.solve()

        if solution:
            # 初始化列表来存储每半小时的净功率和计算后的P_total
            net_power_per_half_hour = []
            ev_vector = []
            P_total_calculated = []  # 用于存储计算后的P_total
            v2g_charge_values = []  # 存储v2g_charge的值
            v2g_discharge_values = []  # 存储v2g_discharge的值
            total_slcak_list = []

            for t in range(self.N_SLOTS):
                # v2g 循环次数
                v2g_charge_values.append(solution.get_value(v2g_charge[t]))
                v2g_discharge_values.append(solution.get_value(v2g_discharge[t]))
                # 计算每个时段的充电功率总和
                charging_power = solution.get_value(slow_charge[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                discharging_power = solution.get_value(slow_discharge[t]) * self.P_slow * self.efficiency
                # 计算每个时段的充电功率总和
                v2g_charging_power = solution.get_value(v2g_charge[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                v2g_discharging_power = solution.get_value(v2g_discharge[t]) * self.P_slow * self.efficiency
                # 计算slack总和
                slack_power = solution.get_value(slack_EV_work[t]) / self.efficiency
                # 计算EV净功率：充电功率 - 放电功率 # 没有v2g
                net_power = (charging_power - discharging_power + v2g_charging_power - v2g_discharging_power
                             + fast_vehicles_distribution[t] * self.P_quick / self.efficiency)
                ev_power = charging_power - discharging_power
                net_power_per_half_hour.append(net_power)
                ev_vector.append(ev_power)

                # 同时计算这个时间段的P_total
                P_total_this_slot = office_P_BASIC[t] + (charging_power - discharging_power) + \
                                    fast_vehicles_distribution[t] * self.P_quick / self.efficiency
                P_total_calculated.append(P_total_this_slot)
                total_slcak_list.append(slack_power)

            # 返回计算好的net_power_per_half_hour和P_total_calculated
            return ev_vector, net_power_per_half_hour, P_total_calculated, total_slcak_list, v2g_charge_values, v2g_discharge_values
        else:
            print(" Office EVLOAD No solution found.")

    def optimizeOfficeChargingV2GPattern(self, fast_vehicles_distribution, office_P_BASIC, not_charging_distribution,
                                         leaving_not_charging):
        model = Model("OfficeChargingV2GPattern")

        # 决策变量
        # 慢充车辆充电状态
        MAX_EV = max(not_charging_distribution)
        state_charge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="charge")
        state_discharge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="discharge")
        # 为每个时段t创建一个二进制辅助变量
        z = model.binary_var_list(self.N_SLOTS, name="z")

        # 约束条件
        constraints = []

        # 约束0 不能同时充放电
        for t in range(self.N_SLOTS):
            # state_charge[t] 可以大于0当且仅当 z[t] == 1
            model.add_constraint(state_charge[t] <= MAX_EV * z[t])
            # state_discharge[t] 可以大于0当且仅当 z[t] == 0
            model.add_constraint(state_discharge[t] <= MAX_EV * (1 - z[t]))

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        for t in range(self.N_SLOTS):
            # 充电或放电的v2g车辆数不超过该时段车辆数
            constraints.append(model.sum(state_charge[t] + state_discharge[t])
                               <= not_charging_distribution[t])

        # 约束2：紧急需求车辆的充电要求
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            still_parking_EV = not_charging_distribution[t] - leaving_not_charging[t]
            net_charging = model.sum((state_charge[t_prime] - state_discharge[t_prime]) * self.P_slow * self.DELTA_T
                                        for t_prime in range(t + 1)) # 不平衡的车
            constraints.append(net_charging <= still_parking_EV * self.CAP_BAT_EV/0.6 * 0.1) #如果是正数，最多充0.1 SOC
            constraints.append(net_charging >= - still_parking_EV * self.CAP_BAT_EV/0.6 * 0.3) # 如果是负数，最多放0.3 SOC

        # 约束3：v2g 充电-放电 = 0
        # 慢充车辆约束
        total_charging_provided = model.sum(state_charge[t]
                                            for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(state_discharge[t]
                                              for t in range(self.N_SLOTS))
        # 确保总充电量=0
        constraints.append(total_charging_provided - total_discharging_reduced == 0)


        # 约束4：累计充电量不超过之前累计的放电量
        for t in range(self.N_SLOTS):
            cumulative_charge_until_t = model.sum(state_charge[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t + 1))  # 包括当前时段
            cumulative_discharge_until_t = model.sum(state_discharge[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(cumulative_discharge_until_t >= cumulative_charge_until_t)
            # 放电量限制
            # cumulative_leaving_t = np.sum(leaving_not_charging[0:t])
            constraints.append(cumulative_discharge_until_t - cumulative_charge_until_t
                               <= (not_charging_distribution[t]) * self.CAP_BAT_EV/0.6*0.3)

        model.add_constraints(constraints)
        # 目标函数
        # 在目标函数中加入惩罚项，惩罚未满足的紧急充电需求
        # penalty_weight = 1000  # 惩罚权重，根据问题规模和重要性调整
        # penalty_term = model.sum(slack_EV_state[t] * penalty_weight for t in range(self.N_SLOTS))
        # 计算每个时段的电网总负载，考虑慢充充电、慢充放电和快充充电
        penalty_v2g = 0.01  # v2g惩罚权重，尽量小
        penalty_term = ( model.sum(state_charge[t] * penalty_v2g for t in range(self.N_SLOTS))
                        + model.sum(state_discharge[t] * penalty_v2g for t in range(self.N_SLOTS)))

        P_total = [office_P_BASIC[t] +
                   (state_charge[t] * self.P_slow * self.DELTA_T / self.efficiency
                    - state_discharge[t] * self.P_slow * self.DELTA_T * self.efficiency)
                   + fast_vehicles_distribution[t] * self.P_quick * self.DELTA_T / self.efficiency
                   for t in range(self.N_SLOTS)]

        model.minimize(model.max(P_total) - model.min(P_total) + penalty_term)
        # 设置求解时间限制为120秒
        model.parameters.timelimit.set(120)


        solution = model.solve()

        if solution:
            # 初始化列表来存储每半小时的净功率和总电网负载
            ev_vector = []
            net_power_per_half_hour = []
            P_total_calculated = []  # 用于存储计算后的P_total
            total_slcak_list = []
            N_EV = 0
            v2g_charge_values = []  # 存储v2g_charge的值
            v2g_discharge_values = []  # 存储v2g_discharge的值

            for t in range(self.N_SLOTS):
                # v2g 循环次数
                v2g_charge_values.append(solution.get_value(state_charge[t]))
                v2g_discharge_values.append(solution.get_value(state_discharge[t]))

                # 计算每个时段的充电功率总和
                charging_power = solution.get_value(state_charge[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                discharging_power = solution.get_value(state_discharge[t]) * self.P_slow * self.efficiency
                # 计算EV净功率：充电功率 - 放电功率 没有v2g
                net_power = charging_power - discharging_power + fast_vehicles_distribution[t] * self.P_quick / self.efficiency
                ev_power = 0
                net_power_per_half_hour.append(net_power)
                ev_vector.append(ev_power)

                # 同时计算P_total
                P_total_this_slot = office_P_BASIC[t] + (charging_power - discharging_power) + \
                                    fast_vehicles_distribution[t] * self.P_quick / self.efficiency
                P_total_calculated.append(P_total_this_slot)
                total_slcak_list.append(0)

                N_EV = N_EV + solution.get_value(state_charge[t]) - solution.get_value(state_discharge[t])

            # 返回计算好的net_power_per_half_hour和P_total_calculated
            return ev_vector, net_power_per_half_hour, P_total_calculated, total_slcak_list,v2g_charge_values,v2g_discharge_values

        else:
            print(" Office EVLOAD No solution found.")

    def optimizeCommunityChargingV2GPattern(self, community_P_BASIC, not_charging_distribution, leaving_not_charging):
        model = Model("CommunityChargingV2GPattern")

        # 决策变量
        # 慢充车辆充电状态
        MAX_EV = max(not_charging_distribution)
        state_charge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="charge")
        state_discharge = model.integer_var_list(self.N_SLOTS, lb=0, ub=MAX_EV, name="discharge")
        # 为每个时段t创建一个二进制辅助变量
        z_home = model.binary_var_list(self.N_SLOTS, name="z_home")
        # 引入非负松弛变量,没有完全满足的对每个时段t
        # slack_EV_state = {t: model.integer_var(name=f"slack_{t}", lb=0) for t in range(self.N_SLOTS)}

        # 约束条件
        constraints = []

        # 约束0 不能同时充放电
        for t in range(self.N_SLOTS):
            # state_charge[t] 可以大于0当且仅当 z[t] == 1
            model.add_constraint(state_charge[t] <= MAX_EV * z_home[t])
            # state_discharge[t] 可以大于0当且仅当 z[t] == 0
            model.add_constraint(state_discharge[t] <= MAX_EV * (1 - z_home[t]))

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        for t in range(self.N_SLOTS):
            # 充电或放电的v2g车辆数不超过该时段车辆数
            constraints.append(model.sum(state_charge[t] + state_discharge[t])
                               <= not_charging_distribution[t])

        # 约束2：紧急需求车辆的充电要求
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            still_parking_EV = not_charging_distribution[t] - leaving_not_charging[t]
            net_charging = model.sum((state_charge[t_prime] - state_discharge[t_prime]) * self.P_slow * self.DELTA_T
                                     for t_prime in range(t + 1))  # 不平衡的车
            constraints.append(net_charging <= still_parking_EV * self.CAP_BAT_EV * 0.1)  # 如果是正数，最多充0.1 SOC
            constraints.append(net_charging >= - still_parking_EV * self.CAP_BAT_EV * 0.3)  # 如果是负数，最多放0.3 SOC
            # constraints.append(slack_EV_state[t] <= still_parking_EV) #

        # 约束3：v2g 充电-放电 = 0
        # 慢充车辆约束
        total_charging_provided = model.sum(state_charge[t] for t in range(self.N_SLOTS))
        total_discharging_reduced = model.sum(state_discharge[t] for t in range(self.N_SLOTS))
        # total_slcak = model.sum(slack_EV_state[t] for t in range(self.N_SLOTS)) # 没满足的功率
        # 确保总充电量（考虑放电减少的量）满足总需求
        constraints.append(total_charging_provided - total_discharging_reduced == 0)

        # 约束4：累计充电量不超过之前累计的放电量
        for t in range(self.N_SLOTS):
            cumulative_charge_until_t = model.sum(state_charge[t_prime] * self.P_slow * self.DELTA_T
                                                  for t_prime in range(t + 1))  # 包括当前时段
            cumulative_discharge_until_t = model.sum(state_discharge[t_prime] * self.P_slow * self.DELTA_T
                                                     for t_prime in range(t + 1))  # 包括当前时段
            constraints.append(cumulative_discharge_until_t >= cumulative_charge_until_t)
            # 充电量限制
            cumulative_leaving_t = np.sum(leaving_not_charging[0:t])
            constraints.append(cumulative_discharge_until_t - cumulative_charge_until_t
                               <= (not_charging_distribution[t] + cumulative_leaving_t) * self.CAP_BAT_EV / 0.6 * 0.3)

        model.add_constraints(constraints)

        # 目标函数
        # 在目标函数中加入惩罚项，惩罚未满足的紧急充电需求
        # penalty_weight = 1000  # 惩罚权重，根据问题规模和重要性调整
        # penalty_term = model.sum(slack_EV_state[t] * penalty_weight for t in range(self.N_SLOTS))
        # 计算每个时段的电网总负载，考虑慢充充电、慢充放电和快充充电
        P_total = [community_P_BASIC[t] +
                   (state_charge[t] * self.P_slow * self.DELTA_T / self.efficiency
                    - state_discharge[t] * self.P_slow * self.DELTA_T * self.efficiency)
                   for t in range(self.N_SLOTS)]

        model.minimize(model.max(P_total) - model.min(P_total))
        # 设置求解时间限制为120秒
        model.parameters.timelimit.set(120)

        solution = model.solve()

        if solution:
            # 初始化列表来存储每半小时的净功率和总电网负载
            net_power_per_half_hour = []
            ev_vector = []
            P_total_calculated = []  # 用于存储计算后的P_total
            total_slcak_list = []
            v2g_charge_values = []  # 存储v2g_charge的值
            v2g_discharge_values = []  # 存储v2g_discharge的值

            for t in range(self.N_SLOTS):
                # v2g 循环次数
                v2g_charge_values.append(solution.get_value(state_charge[t]))
                v2g_discharge_values.append(solution.get_value(state_discharge[t]))

                # 计算每个时段的充电功率总和
                charging_power = solution.get_value(state_charge[t]) * self.P_slow / self.efficiency
                # 计算每个时段的放电功率总和
                discharging_power = solution.get_value(state_discharge[t]) * self.P_slow * self.efficiency
                # 计算EV净功率：充电功率 - 放电功率
                net_power = charging_power - discharging_power
                ev_power = 0
                net_power_per_half_hour.append(net_power) #计算电压
                ev_vector.append(ev_power) #计算收益

                # 同时计算P_total
                P_total_this_slot = community_P_BASIC[t] + (charging_power - discharging_power)
                P_total_calculated.append(P_total_this_slot)
                total_slcak_list.append(0)

            # 返回计算好的net_power_per_half_hour和P_total_calculated
            return ev_vector, net_power_per_half_hour, P_total_calculated, total_slcak_list, v2g_charge_values, v2g_discharge_values

        else:
            print(" Office EVLOAD No solution found.")

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
    for hour in range(48):
        load_matrix = nodedata_dict[hour]
        re_matrix = re_capacity_dict.get(hour, np.zeros_like(load_matrix))

        for node in nodes:
            node_index = np.where(load_matrix[:, 0] == node)[0][0]
            load = load_matrix[node_index, 1]

            # 对于pv和wt，检查节点是否在对应的矩阵中
            if node in re_matrix[:, 0]:
                re_index = np.where(re_matrix[:, 0] == node)[0][0]
                re = re_matrix[re_index, 1]

            else:
                re = 0

            net_load = load - re

            if node not in P_basic_dict:
                P_basic_dict[node] = [net_load]
            else:
                P_basic_dict[node].extend([net_load])

    return P_basic_dict


def P_basic_and_EV(P_basic_dict, work_slow_charging_distribution, home_charging_distribution,
                   work_quick_charging_distribution, work_slow_arriving, home_arriving):
    CAP_BAT_EV = 42  # 固定的每辆车充电需求（70kWh*0.6）
    DELTA_T = 0.5  # 每个时间段长度，30分钟
    N_SLOTS = 48  # 一天中的时间段数量
    P_slow = 7  # kW
    P_quick = 42  # kW
    efficiency = 1
    # 初始化存储每个节点包括EV负载后的总负载的字典
    node_P_basic_and_EV = {}

    for node in nodes:
        P_basic = P_basic_dict.get(node, [0] * N_SLOTS)  # 获取基础负载
        EV_load = [0] * N_SLOTS  # 初始化EV负载列表

        if 100 <= node < 199:  # Office节点
            slow_charging_slots = work_slow_charging_distribution.get(node, [0] * N_SLOTS)
            quick_charging_slots = work_quick_charging_distribution.get(node, [0] * N_SLOTS)
            arriving_slots = work_slow_arriving.get(node, [0] * (N_SLOTS-1))

            # 快充车辆充电计算
            for t in range(N_SLOTS):
                EV_load[t] += quick_charging_slots[t] * P_quick / efficiency

            # 0时刻的车辆充电处理
            for i in range(0, 12):
                if i < N_SLOTS:
                    EV_load[i] += slow_charging_slots[0] * P_slow / efficiency

            # 从1时刻开始使用arriving_slots
            for t in range(1, N_SLOTS):
                if t-1 < len(arriving_slots):
                    arriving_cars = arriving_slots[t-1]
                    for i in range(t, min(t + 12, N_SLOTS)):
                        EV_load[i] += arriving_cars * P_slow / efficiency
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

            # 从1时刻开始使用arriving_slots
            for t in range(1, N_SLOTS):
                if t-1 < len(arriving_slots):
                    arriving_cars = arriving_slots[t-1]
                    for i in range(t, min(t + 12, N_SLOTS)):
                        EV_load[i] += arriving_cars * P_slow  / efficiency
                        #检查负载是否够了
                        if np.sum(EV_load[:i + 1]) >= sum(arriving_slots) * CAP_BAT_EV:
                            break
                if np.sum(EV_load) >= sum(arriving_slots) * CAP_BAT_EV:
                    break

        # 将EV负载与基础负载相加得到总负载
        P_total_with_EV = [P_basic[t] + EV_load[t] for t in range(N_SLOTS)]
        node_P_basic_and_EV[node] = P_total_with_EV

    return node_P_basic_and_EV


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
    # 初始化存储每半小时所有节点EV负荷的字典
    node_EV_load = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时所有节点EV slow负荷的字典
    node_EV_slowload = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时所有节点快充EV负荷的字典
    node_quick_EV_load = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时所有节点slack的字典
    node_slack_load = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储Ptotal的字典
    node_P_total = {}
    # 初始化存储V2G充电的字典
    node_P_V2G_charge = {}
    node_P_V2G_discharge = {}
    # 初始化存储每个微电网48步长EV负荷的字典
    mic_EV_load = {mic: np.zeros(48) for mic in range(4)}  # 4个微电网
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
    node_P_basic_and_EV = P_basic_and_EV(P_basic_dict, work_slow_charging_distribution, home_charging_distribution,
                                         work_quick_charging_distribution,work_slow_arriving,home_arriving)

    # 更新快充负荷数据
    for node, quick_load_vector in work_quick_charging_distribution.items():
        # 获取当前节点的索引
        node_idx = node_mapping[node]
        P_quick = quick_load_vector * 42 * 0.5
        for t in range(48):
            node_quick_EV_load[t][node_idx] = P_quick[t]

    #优化实例
    optimize = EVChargingOptimizer()
    for node in tqdm(nodes, desc="Processing nodes", ncols=100, unit="node"):
        # 获取当前节点的索引
        node_idx = node_mapping[node]

        # Office节点
        if 100 <= node < 199:
            P_basic = P_basic_dict[node]
            if max(work_slow_charging_distribution.get(node, [0])) == 0:
                ev_load_vector, net_power, P_total_calculated, total_slcak, v2g_charge_values, v2g_discharge_values\
                    = optimize.optimizeOfficeChargingV2GPattern(
                    fast_vehicles_distribution=work_quick_charging_distribution.get(node, []),
                    office_P_BASIC=P_basic,
                    not_charging_distribution=not_charging_distribution.get(node, []),
                    leaving_not_charging=not_charging_leaving.get(node, []))
                # # 只有快充，直接计算
                # net_power_per_half_hour = []
                # for t1 in range(48):
                #     # 计算EV净功率
                #     net_power = (work_quick_charging_distribution.get(node, [0])[t1] * 42 / 0.9)  # 快充
                #     # 将时间段和净功率添加到字典中
                #     net_power_per_half_hour.append(net_power)
                #     ev_load_vector = net_power_per_half_hour
            else: # 有慢充
                # 进行优化
                ev_load_vector, net_power, P_total_calculated, total_slcak, v2g_charge_values, v2g_discharge_values\
                    = optimize.optimizeOfficeChargingPattern(
                    slow_vehicles_distribution=work_slow_charging_distribution.get(node, []),
                    slow_arriving_vehicles=work_slow_arriving.get(node, []),
                    slow_leaving_vehicles=work_slow_leaving.get(node, []),
                    fast_vehicles_distribution=work_quick_charging_distribution.get(node, []),
                    office_P_BASIC=P_basic,
                    not_charging_distribution=not_charging_distribution.get(node, []),
                    leaving_not_charging=not_charging_leaving.get(node, [])
                )
        else:  # 社区节点
            P_basic = P_basic_dict[node]
            if max(home_charging_distribution.get(node, [0])) == 0:
                # # 如果没有社区车辆，跳过优化步骤
                # ev_load_vector = np.zeros(48)  # 48个半小时时段
                # P_total_calculated = P_basic
                ev_load_vector, net_power, P_total_calculated, total_slcak, v2g_charge_values, v2g_discharge_values\
                    = optimize.optimizeCommunityChargingV2GPattern(
                    community_P_BASIC=P_basic,
                    not_charging_distribution=not_charging_distribution.get(node, []),
                    leaving_not_charging=not_charging_leaving.get(node, [])
                )
            else:
                # 进行优化
                ev_load_vector, net_power, P_total_calculated, total_slcak, v2g_charge_values, v2g_discharge_values\
                    = optimize.optimizeCommunityChargingPattern(
                    community_vehicles_distribution=home_charging_distribution.get(node, []),
                    community_arriving_vehicles=home_arriving.get(node, []),
                    community_leaving_vehicles=home_leaving.get(node, []),
                    community_P_BASIC=P_basic,
                    not_charging_distribution=not_charging_distribution.get(node, []),
                    leaving_not_charging=not_charging_leaving.get(node, [])
                )

        # 使用该节点的EV负荷更新node_EV_load字典中的向量
        node_P_total[node] = P_total_calculated
        node_slack_load[node] = total_slcak
        node_P_V2G_charge[node] = v2g_charge_values
        node_P_V2G_discharge[node] = v2g_discharge_values
        save_dict_to_csv(base_path, node_P_V2G_charge, 'node_P_V2G_charge')
        save_dict_to_csv(base_path, node_P_V2G_discharge, 'node_P_V2G_discharge')

        for t in range(48):
            node_EV_load[t][node_idx] = net_power[t]
            node_EV_slowload[t][node_idx] = ev_load_vector[t]



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
                mic_EV_load[mic_idx][t] += node_EV_slowload[t][node_idx]
                mic_EV_load_quick[mic_idx][t] += node_quick_EV_load[t][node_idx]
    print("EV计算结束")
    return node_EV_load, mic_EV_load, node_P_total, node_P_basic_and_EV, mic_EV_load_quick, node_slack_load, P_basic_dict

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
# #
# # #实例调用
# EV_Q1 = np.array(prices_real)
# EV_S1 = np.array(prices_real)
# EV_2 = np.array(prices_real)
# EV_3 = np.array(prices_real)
# EV_4 = np.array(prices_real)
#
# EV_p = 800 * 0.9
# v2g = 0.5
#
# node_EV_load, mic_EV_load, node_P_total, node_P_basic_and_EV,mic_EV_load_quick, node_slack_load, P_basic_dict \
#     = EVload(EV_Q1, EV_S1, EV_2, EV_3, EV_4,EV_p,v2g,'data2', nodedata_dict, re_capacity_dict)
#
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