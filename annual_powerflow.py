from gridinfo import (branch, bus, gen, reverse_node_mapping, charge_ratio)
import warnings
#from data_output import EV_penetration, v2g_ratio
# import numpy as np
import pandas as pd
import os
import pandapower as pp
# import pandapower.timeseries as ts
# 忽略特定库的警告
warnings.filterwarnings("ignore")

class OPF:
    def __init__(self, nodedata_dict, re_capacity_dict):
        self.net = pp.create_empty_network()
        self.gen_cost = {101: (0.02801, 25.8027, 24.6382),
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
                          p_mw=0,  # 有功功率
                          min_q_mvar=gen['Qmin'][idx] / 1000.0,
                          max_q_mvar=gen['Qmax'][idx] / 1000.0,
                          vm_pu=1.0,
                          min_p_mw=0.0,
                          max_p_mw=gen['Pmax'][idx] / 1000.0,
                          type='thermal'  # 标记为火力发电
                          )


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
            if idx > 10:
                continue  # 跳过索引大于10的发电机
            # 获取发电机的母线索引
            bus_index = gen.bus
            # 从pandapower网络中检索母线名称
            bus_name = self.net.bus.loc[bus_index, 'name']
            bus_name_int = int(bus_name)

            # 检查发电机的类型
            if gen['type'] == 'thermal':  # 仅为火力发电添加成本
                # 从gen_cost字典获取成本函数参数
                cost_params = self.gen_cost[bus_name_int]
                # 使用发电机的索引添加成本函数
                pp.create_poly_cost(self.net, element=idx, et="gen",
                                    cp1_eur_per_mw=cost_params[1],  # 线性项
                                    cp2_eur_per_mw2=cost_params[0],  # 二次项
                                    cp0_eur=cost_params[2])  # 常数项

    # 添加可再生能源和负荷
    def add_loads_and_sgen_for_each_period(self, period):
        #period = int(i / 2)  # 48转24h对应
        # 清除先前的负荷和分布式发电
        self.net.load.drop(self.net.load.index, inplace=True)
        self.net.sgen.drop(self.net.sgen.index, inplace=True)

        # # 添加风光发电的有功功率
        # for node_name, p_mw in self.re_capacity_dict.get(period, []):
        #     bus_index = self.net.bus.index[self.net.bus.name == str(int(node_name))].item()
        #     pp.create_sgen(self.net, bus=bus_index, p_mw=p_mw / 1000.0)  # kW to MW

        # 添加风光发电的有功功率（使用动态发电机）
        for node_name, p_mw in self.re_capacity_dict.get(period, []):
            bus_index = self.net.bus.index[self.net.bus.name == str(int(node_name))].item()

            pp.create_gen(self.net, bus=bus_index,
                          p_mw=0,  # 有功功率
                          min_q_mvar=0,
                          max_q_mvar=0,
                          vm_pu=1.0,
                          min_p_mw=0.0,
                          max_p_mw=p_mw / 1000.0,
                          type='renewable'  # 标记为风光发电
                          )
            #
            # print("Generated generator:", gen)  # 打印发电机对象
            # print("Current generators in the network:", self.net.gen)
            # # 设置成本为0
            # pp.create_poly_cost(self.net, element=gen, et="gen", cp1_eur_per_mw=0.0)  # 设定成本属性

        # 添加常规负荷的有功和无功功率
        for node_name, p_mw, q_mvar in self.nodedata_dict.get(period, []):
            node_name = int(node_name)
            bus_index = self.net.bus.index[self.net.bus.name == str(int(node_name))].item()
            pp.create_load(self.net, bus=bus_index, p_mw=p_mw / 1000.0,
                           q_mvar=q_mvar / 1000.0)  # kW to MW, kVAR to MVAR

    def run_ts_opf(self, EV_penetration, v2g_ratio, file_path):

        self.build_network()
        self.add_branches()
        self.add_generator_costs()

        generator_costs = []  # 存储每个发电机的成本
        system_losses = []  # 存储每个步长的网损
        import_powers = []  # 存储每个步长从输电网进口的电量
        gen_powers = []  # 存储每个发电
        renewable_power = []  # 存储风光弃电量
        renewable_curtailed = []  # 存储风光弃电量

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
            self.net.line["max_loading_percent"] = 100  # 载流量约束
            # all_generators = self.net.gen
            # print(all_generators)

            pp.runopp(self.net, verbose=False)  # 执行最优潮流计算
            # print("OPF result:", self.net.res_cost)

            # 将当前时段的电压向量添加到DataFrame中
            current_voltage_vector = self.net.res_bus[['vm_pu']].rename(columns={'vm_pu': f'period_{period}'})
            voltage_distributions = pd.concat([voltage_distributions, current_voltage_vector], axis=1)

            # # 计算并存储每个发电机的成本
            # gen_costs = self.calculate_generator_costs()
            # generator_costs.append(gen_costs)

            # 计算网损
            loss_mw = sum(self.net.res_line.pl_mw)
            system_losses.append(loss_mw/1000)

            # 计算从输电网进口的电量
            import_power = sum(self.net.res_ext_grid.p_mw)  # 直接从res_ext_grid获取外部电网注入的有功功率
            import_powers.append(import_power/1000)

            total_cost = self.net.res_cost  # 获取所有成本的总和
            generator_costs.append(total_cost)

            # 计算风光发电的理论输出和实际输出
            # for idx, gen in self.net.gen.iterrows():  # 遍历所有发电机
            #     if gen.get('type') == 'renewable':  # 检查发电机类型
            #         bus_index = gen.bus  # 获取发电机对应的母线索引
            #         bus_name = self.net.bus.loc[bus_index, 'name']  # 获取母线名称
            #         # 从 res_gen 获取当前发电机的输出功率
            #         renewable_output = self.net.res_gen.loc[idx, 'p_mw']
            #         print(f"Renewable generator at bus {bus_name}: {renewable_output} MW")

            # print(f"Period {period} renewable capacity data:")
            # renewable_data = self.re_capacity_dict.get(period, [])
            # print(renewable_data)

            renewable_expected = sum(p_mw / 1000.0 for _, p_mw in self.re_capacity_dict.get(period, []))
            renewable_actual = sum(
                self.net.res_gen.loc[idx, 'p_mw'] for idx, gen in self.net.gen.iterrows() if
                gen.get('type') == 'renewable'
            )
            # print(renewable_expected * 1000)
            # print(renewable_actual)
            renewable_curtailed.append(renewable_expected - renewable_actual/1000)  # 计算风光弃电kW
            renewable_power.append(renewable_actual/1000)

            thermal_actual = sum(
                self.net.res_gen.loc[idx, 'p_mw'] for idx, gen in self.net.gen.iterrows() if
                gen.get('type') == 'thermal'
            )
            gen_powers.append(thermal_actual/1000)

        # 所有时段处理完成后，保存电压分布到单个CSV文件
        voltage_distribution_file = os.path.join(base_path, 'voltage_distribution.csv')
        voltage_distributions.to_csv(voltage_distribution_file, index=False)

        return generator_costs, system_losses, import_powers, gen_powers, renewable_curtailed,renewable_power

    # def calculate_generator_costs(self):
    #     gen_costs = []
    #     for idx, row in enumerate(self.net.gen.iterrows()):
    #         gen_id = row[1]['bus']  # 获取发电机对应的母线ID
    #         p_mw = self.net.res_gen.loc[idx, "p_mw"]  # 获取发电机的有功功率输出
    #         # 根据母线索引从 self.net.bus DataFrame 中检索母线名称
    #         bus_name = self.net.bus.loc[gen_id, 'name']
    #         bus_name_int = int(bus_name)
    #         if bus_name_int in self.gen_cost:  # 检查发电机ID是否在成本参数字典中
    #             cost_params = self.gen_cost[bus_name_int]  # 获取成本函数参数
    #             a = cost_params[0]
    #             b = cost_params[1]
    #             c = cost_params[2]
    #             cost = a * p_mw ** 2 + b * p_mw + c  # 计算成本
    #             gen_costs.append((gen_id, cost))  # 将发电机ID和计算得到的成本添加到列表中
    #         else:
    #             gen_costs.append((gen_id, 0))  # 如果发电机ID不在字典中，添加None作为成本
    #     return gen_costs


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





