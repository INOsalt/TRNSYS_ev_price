from gridinfo import (branch, bus, gen, reverse_node_mapping, charge_ratio)
import warnings
#from data_output import EV_penetration, v2g_ratio
import numpy as np
import pandas as pd
import os
import pandapower as pp
import pandapower.timeseries as ts
# 忽略特定库的警告
warnings.filterwarnings("ignore")

class OPF:
    def __init__(self, node_EV_load, nodedata_dict, re_capacity_dict):
        self.net = pp.create_empty_network()
        self.node_EV_load = node_EV_load
        self.gen_cost = {101: (0.01199, 37.5510, 117.7551),
                    201: (0.02801, 25.8027, 24.6382),
                    301: (0.02801, 25.8027, 24.6382),
                    401: (0.02801, 25.8027, 24.6382)}
        self.nodedata_dict = nodedata_dict
        self.re_capacity_dict = re_capacity_dict
    def build_network(self):
        # 添加母线
        for idx, row in enumerate(bus['bus_i']):
            pp.create_bus(self.net, vn_kv=bus['baseKV'][idx], name=str(row),
                          max_vm_pu=bus['Vmax'][idx], min_vm_pu=bus['Vmin'][idx],
                          type=bus['type'][idx])

        # 添加发电机
        for idx, row in enumerate(gen['gen_bus']):
            # 在母线DataFrame中查找与母线名称匹配的索引
            gen_bus_name = str(gen['gen_bus'][idx])
            bus_index = self.net.bus.index[self.net.bus.name == gen_bus_name].item()
            # 创建发电机时包括最小和最大有功功率限制
            pp.create_gen(self.net, bus=bus_index,
                          p_mw=0,  # 将有功功率从kW转换为MW
                          min_q_mvar=gen['Qmin'][idx] / 1000.0,  # 将无功功率下限从kVAR转换为MVAR
                          max_q_mvar=gen['Qmax'][idx] / 1000.0,  # 将无功功率上限从kVAR转换为MVAR
                          vm_pu=1.0,  # 假设目标电压为1.0pu
                          min_p_mw=0.0,  # 假设这里是最小有功功率限制，需要从kW转换为MW
                          max_p_mw=gen['Pg'][idx] / 1000.0)


        # 创建slack
        # bus_index_101是编号为101的母线在pandapower网络中的索引
        bus_index_101 = self.net.bus.index[self.net.bus.name == '101'].item()
        # 创建外部电网并连接到母线101，同时设置为slack母线
        pp.create_ext_grid(self.net, bus=bus_index_101, vm_pu=1.00, va_degree=0, slack=True)

    def add_branches(self):
        for idx, row in enumerate(branch['fbus']):
            if branch['status'][idx] == 1:  # 检查支路是否闭合
                # 将浮点数母线名称转换为整数字符串
                from_bus_name = str(int(branch['fbus'][idx]))
                from_bus_index = self.net.bus.index[self.net.bus.name == from_bus_name].item()
                # 在母线DataFrame中查找与母线名称匹配的'tbus'索引
                to_bus_name = str(int(branch['tbus'][idx]))
                to_bus_index = self.net.bus.index[self.net.bus.name == to_bus_name].item()

                # # 计算最大负载百分比
                # I_rated = branch['Normalrating'][idx] / 1000  # 线路的额定电流
                # I_max = branch['SCCR'][idx]  # 线路的热极限电流，单位 kA
                # max_loading_percent = (I_rated / I_max) * 100  # 计算最大负载百分比

                pp.create_line_from_parameters(self.net,
                                               from_bus=from_bus_index,
                                               to_bus=to_bus_index,
                                               length_km=branch['Length'][idx],
                                               r_ohm_per_km=branch['r'][idx],
                                               x_ohm_per_km=branch['x'][idx],
                                               c_nf_per_km=0,  # 可根据实际情况调整
                                               max_i_ka=(branch['Normalrating'][idx]/1000),
                                               #std_type="custom_line"
                                               ) # 使用自定义类型以便能够设置 max_loading_percent
                # # 更新线路的最大负载百分比
                # self.net.line.at[line_idx, 'max_loading_percent'] = max_loading_percent
    def add_generator_costs(self):
        # 为每个发电机添加成本函数
        for idx, gen in self.net.gen.iterrows():
            # 使用 gen.bus 来获取母线索引
            bus_index = gen.bus
            # 根据母线索引从 self.net.bus DataFrame 中检索母线名称
            bus_name = self.net.bus.loc[bus_index, 'name']
            bus_name_int = int(bus_name)
            # 从gen_cost字典获取成本函数参数
            cost_params = self.gen_cost[bus_name_int]
            # 使用发电机的索引（idx）作为element参数来添加成本函数
            pp.create_poly_cost(self.net, element=idx, et="gen",
                                cp1_eur_per_mw=cost_params[1],  # b参数
                                cp2_eur_per_mw2=cost_params[0],  # a参数
                                cp0_eur=cost_params[2])  # c参数

    # 添加可再生能源和负荷
    def add_loads_and_sgen_for_each_period(self, period):
        #period = int(i / 2)  # 48转24h对应
        # 清除先前的负荷和分布式发电
        self.net.load.drop(self.net.load.index, inplace=True)
        self.net.sgen.drop(self.net.sgen.index, inplace=True)

        # 获取当前时段的所有节点负荷
        loads_for_period = self.node_EV_load[period]  # This should be a vector of length 40

        # 添加每个节点的电动汽车负荷
        for node_index, load_kw in enumerate(loads_for_period):
            # 使用reverse_node_mapping获取节点名称
            node_name = reverse_node_mapping[node_index]
            # 在pandapower中找到对应的母线索引
            bus_index = self.net.bus.index[self.net.bus.name == str(node_name)].item()
            # 添加负荷
            pp.create_load(self.net, bus=bus_index, p_mw=load_kw / 1000.0)  # kW to MW

        # 添加风光发电的有功功率
        for node_name, p_mw in self.re_capacity_dict.get(period, []):
            bus_index = self.net.bus.index[self.net.bus.name == str(int(node_name))].item()
            pp.create_sgen(self.net, bus=bus_index, p_mw=p_mw / 1000.0)  # kW to MW

        # 添加常规负荷的有功和无功功率
        for node_name, p_mw, q_mvar in self.nodedata_dict.get(period, []):
            node_name = int(node_name)
            bus_index = self.net.bus.index[self.net.bus.name == str(int(node_name))].item()
            pp.create_load(self.net, bus=bus_index, p_mw=p_mw / 1000.0,
                           q_mvar=q_mvar / 1000.0)  # kW to MW, kVAR to MVAR

    # def add_pv_wt_with_q_compensation(self):
    #     # 遍历pvwt_reactive字典中的'pvwt_bus'数组
    #     for idx, bus in enumerate(pvwt_reactive['pvwt_bus']):
    #         # 获取无功功率上下限
    #         q_max = pvwt_reactive['Qmax'][idx] / 1000.0  # 转换为MVAR
    #         q_min = pvwt_reactive['Qmin'][idx] / 1000.0  # 转换为MVAR
    #
    #         # 在pandapower网络中查找对应的母线索引
    #         bus_index = self.net.bus.index[self.net.bus.name == str(bus)].item()
    #
    #         # 创建无功发生器
    #         pp.create_sgen(self.net, bus=bus_index, p_mw=0,
    #                        q_mvar=0,  # 初始无功功率设置为0
    #                        min_q_mvar=q_min,
    #                        max_q_mvar=q_max,
    #                        name=f"PV_WT_{bus}")

    def run_ts_opf(self, EV_penetration, v2g_ratio, file_path):

        self.build_network()
        self.add_branches()
        self.add_generator_costs()

        generator_costs = []  # 存储每个发电机的成本
        system_losses = []  # 存储每个步长的网损
        import_powers = []  # 存储每个步长从输电网进口的电量
        gen_powers = []

        # 定义文件保存路径
        base_path = os.path.join(file_path, str(EV_penetration), str(v2g_ratio))
        if not os.path.exists(base_path):
            os.makedirs(base_path)  # 如果路径不存在，创建对应的文件夹

        # 初始化DataFrame来收集所有时段的电压向量
        voltage_distributions = pd.DataFrame()

        # 对于每个半小时的时间段执行OPF
        for period in range(48):
            self.add_loads_and_sgen_for_each_period(period)
            # self.add_pv_wt_with_q_compensation()
            self.net.line["max_loading_percent"] = 90  # 载流量约束

            pp.runopp(self.net, verbose=True)  # 执行最优潮流计算

            # 将当前时段的电压向量添加到DataFrame中
            current_voltage_vector = self.net.res_bus[['vm_pu']].rename(columns={'vm_pu': f'period_{period}'})
            voltage_distributions = pd.concat([voltage_distributions, current_voltage_vector], axis=1)

            # 计算并存储每个发电机的成本
            gen_costs = self.calculate_generator_costs()
            generator_costs.append(gen_costs)

            # 计算网损
            loss_mw = sum(self.net.res_line.pl_mw)
            system_losses.append(loss_mw)

            # 计算从输电网进口的电量
            import_power = sum(self.net.res_ext_grid.p_mw)  # 直接从res_ext_grid获取外部电网注入的有功功率
            import_powers.append(import_power)

            # 计算发电的电量
            total_gen_power = sum(self.net.res_gen.p_mw)
            gen_powers.append(total_gen_power)

        # 所有时段处理完成后，保存电压分布到单个CSV文件
        voltage_distribution_file = os.path.join(base_path, 'voltage_distribution.csv')
        voltage_distributions.to_csv(voltage_distribution_file, index=False)

        return generator_costs, system_losses, import_powers, gen_powers

    def calculate_generator_costs(self):
        gen_costs = []
        for idx, row in enumerate(self.net.gen.iterrows()):
            gen_id = row[1]['bus']  # 获取发电机对应的母线ID
            p_mw = self.net.res_gen.loc[idx, "p_mw"]  # 获取发电机的有功功率输出
            # 根据母线索引从 self.net.bus DataFrame 中检索母线名称
            bus_name = self.net.bus.loc[gen_id, 'name']
            bus_name_int = int(bus_name)
            if bus_name_int in self.gen_cost:  # 检查发电机ID是否在成本参数字典中
                cost_params = self.gen_cost[bus_name_int]  # 获取成本函数参数
                a = cost_params[0]
                b = cost_params[1]
                c = cost_params[2]
                cost = a * p_mw ** 2 + b * p_mw + c  # 计算成本
                gen_costs.append((gen_id, cost))  # 将发电机ID和计算得到的成本添加到列表中
            else:
                gen_costs.append((gen_id, 0))  # 如果发电机ID不在字典中，添加None作为成本
        return gen_costs

# EVload_example = {k: [0]*40 for k in range(48)}
# opf = OPF(EVload_example)
# generator_costs, system_losses, import_powers = opf.run_ts_opf()
# print('cost:', generator_costs)
# print('lose',system_losses)
# print('power',import_powers)
# sums = {}
# for sublist in generator_costs:
#     for pair in sublist:
#         key, value = pair
#         if key in sums:
#             sums[key] += value
#         else:
#             sums[key] = value
# print(sums)





