from gridinfo import branch, bus, gen, mapped_nodedata_dict, node_mapping, mapped_pv_dict, mapped_wt_dict
import numpy as np
from PF_NR import PowerFlow

def powerflow(EVload):

    for i in range(48):  # 修正循环，从0到47
        # 获取当前小时的电动汽车负荷
        Pev_i = EVload[i]
        # 更新bus字典中的Pd和Qd
        # 假设nodedata_dict[i]是一个二维数组或类似结构，其中包含了当前小时所有节点的Pd和Qd值
        bus["Pd"] = mapped_nodedata_dict[i][:, 1]  # 更新Pd为nodedata_dict的第2列
        bus["Qd"] = mapped_nodedata_dict[i][:, 2]  # 更新Qd为nodedata_dict的第3列
        bus["Pd"] += Pev_i

        # gen data PV WT
        # 更新pv
        for row in mapped_pv_dict[i]:
            pv_bus, pvi = row  # 第一列是 gen_bus，第二列是 capacity
            # 找到对应 gen_bus 的索引
            index = np.where(gen['gen_bus'] == pv_bus)[0]
            if len(index) > 0:  # 确保找到了匹配的 gen_bus
                # 更新 Pg 值
                gen['Pg'][index] = pvi
        # 更新wt
        for row in mapped_wt_dict[i]:
            wt_bus, wti = row  # 第一列是 gen_bus，第二列是 capacity
            # 找到对应 gen_bus 的索引
            index = np.where(gen['gen_bus'] == wt_bus)[0]
            if len(index) > 0:  # 确保找到了匹配的 gen_bus
                # 更新 Pg 值
                gen['Pg'][index] = wti

        # 初始化status数组，默认所有连接都是激活的，即状态为1
        branch['status'] = np.ones_like(branch['fbus'])

        # 更新微电网间功率，使用节点到索引的映射转换节点对
        # 定义节点对应的微电网关系
        microgrid_relations = [
            (102, 201, 1, 2),
            (104, 301, 1, 3),
            (208, 401, 2, 4),
            (205, 310, 2, 3),
            (318, 404, 3, 4)
        ]
        for source, target, mg_source, mg_target in microgrid_relations:
            # 使用node_mapping来获取源节点和目标节点的索引
            source_index = node_mapping[source]
            target_index = node_mapping[target]
            # 从Pnet_mic获取对应微电网间的功率流
            power_flow = Pnet_mic[i].get((mg_source, mg_target), 0)






