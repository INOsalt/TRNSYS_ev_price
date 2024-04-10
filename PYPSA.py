import pypsa
import numpy as np
import pandas as pd
from gridinfo import (branch, bus, gen, reverse_node_mapping, charge_ratio)
import warnings
import os



class OPF_PyPSA:
    def __init__(self, node_EV_load, nodedata_dict, re_capacity_dict):
        self.net = pypsa.Network()
        self.node_EV_load = node_EV_load
        self.gen_cost = {
            101: (0.01199, 37.5510, 117.7551),
            201: (0.02801, 25.8027, 24.6382),
            301: (0.02801, 25.8027, 24.6382),
            401: (0.02801, 25.8027, 24.6382)
        }
        self.nodedata_dict = nodedata_dict
        self.re_capacity_dict = re_capacity_dict

    def build_network(self):
        # 假设 bus, gen 等为先前定义的 DataFrame 或字典

        # 添加母线
        for idx, row in bus.iterrows():
            self.net.add("Bus", name=str(row['bus_i']), vn_kv=row['baseKV'],
                         type="b" if row['type'] == 3 else "n",  # 在PyPSA中，type "b" 代表平衡节点（slack）
                         max_vm_pu=row['Vmax'], min_vm_pu=row['Vmin'])

        # 添加发电机
        for idx, row in gen.iterrows():
            gen_bus_name = str(row['gen_bus'])
            # 发电机连接到的母线应该是先前添加的母线名称
            self.net.add("Generator", name=f"Gen_{idx}", bus=gen_bus_name,
                         p_nom=row['Pg'] / 1000.0,  # 注意单位转换 kW to MW
                         p_min_pu=row['Qmin'] / row['Pg'],  # PyPSA中使用pu系统
                         p_max_pu=row['Qmax'] / row['Pg'],
                         control="PQ")  # 假定发电机的有功和无功功率都被控制

        # 添加外部电网（Slack）
        # 在PyPSA中，外部电网（ExtGrid）连接到的母线通过控制属性定义其为slack母线
        self.net.add("ExtGrid", name="Slack", bus="101", vm_pu=1.00)

    def add_branches(self):
        for idx in range(len(branch['fbus'])):
            if branch['status'][idx] == 1:  # 确保线路是活动状态
                from_bus_name = f"Bus {int(branch['fbus'][idx])}"
                to_bus_name = f"Bus {int(branch['tbus'][idx])}"

                # 添加线路到PyPSA网络
                self.net.add("Line",
                  name=f"Line {idx}",
                  bus0=from_bus_name,
                  bus1=to_bus_name,
                  r_ohm=branch['r'][idx],  # 直接使用总电阻值
                  x_ohm=branch['x'][idx],  # 直接使用总电抗值
                  s_nom=branch.get('Normalrating', 10.0)[idx] if 'Normalrating' in branch else 10.0)

    def add_generator_costs(self):
        # 遍历所有发电机以添加成本函数
        for idx, gen in self.net.generators.iterrows():
            # 注意：在PyPSA中，发电机和其它元素是通过名字进行引用的
            gen_name = gen.name
            bus_id = gen.bus

            # 检查是否为每个发电机定义了成本参数
            if int(bus_id) in self.gen_cost:
                cost_params = self.gen_cost[int(bus_id)]
                # 为发电机添加多项式成本
                self.net.add("PolyCost", element=gen_name, et="generator",
                             coefficients=np.poly1d([cost_params[0], cost_params[1], cost_params[2]]))

    def preprocess_time_series_data(self):
        # self.node_EV_load, self.re_capacity_dict等变量以一种能够直接转换为PyPSA时间序列格式的方式组织。
        # 在实际中，可能需要将这些数据预处理成适合PyPSA的格式。
        # PyPSA的时间序列通常是pandas.DataFrame，索引是时间点（快照），列名是网络元素的名称。

        # 为每个负荷创建时间序列
        for node_index, loads in enumerate(zip(*self.node_EV_load.values())):
            node_name = reverse_node_mapping[node_index]
            self.net.add("Load", node_name, bus=node_name, p_set=loads)

        # 为每个可再生能源发电单元创建时间序列
        for period, gens in self.re_capacity_dict.items():
            for node_name, p_mw in gens:
                # 假设这里的 p_mw 是每个时间点的功率值列表
                self.net.add("Generator", f"SGEN_{node_name}", bus=str(node_name), p_nom_extendable=True, p_max_pu=p_mw)

        # 添加常规负荷的有功和无功功率时间序列（如果需要）

    def run_ts_opf(self, file_path, EV_penetration, v2g_ratio):

        self.build_network()
        self.add_branches()
        self.add_generator_costs()
        self.preprocess_time_series_data()

        # 执行线性最优潮流计算
        self.net.lopf(self.net.snapshots, solver_name='cbc')

        # 定义文件保存路径
        base_path = os.path.join(file_path, str(EV_penetration), str(v2g_ratio))
        if not os.path.exists(base_path):
            os.makedirs(base_path)  # 如果路径不存在，创建对应的文件夹

        # 保存电压分布
        voltage_distributions = self.net.buses_t.v_mag_pu
        voltage_distribution_file = os.path.join(base_path, 'voltage_distribution.csv')
        voltage_distributions.to_csv(voltage_distribution_file)

        # 保存总成本
        total_cost = self.net.objective
        cost_file = os.path.join(base_path, 'total_cost.txt')
        with open(cost_file, 'w') as f:
            f.write(str(total_cost))

        # 保存系统损失
        system_losses = self.net.lines_t.p0.abs().sum(axis=1).sum()
        loss_file = os.path.join(base_path, 'system_losses.txt')
        with open(loss_file, 'w') as f:
            f.write(str(system_losses))

        # 保存从输电网进口的电量
        import_powers = self.net.ext_grid_t.p_mw.abs().sum(axis=1).sum()
        import_power_file = os.path.join(base_path, 'import_powers.csv')
        import_powers.to_csv(import_power_file)

        # 保存发电量
        gen_powers = self.net.generators_t.p.sum(axis=1).sum()
        gen_powers_file = os.path.join(base_path, 'gen_powers.csv')
        gen_powers.to_csv(gen_powers_file)

        return total_cost, system_losses, import_powers, gen_powers

