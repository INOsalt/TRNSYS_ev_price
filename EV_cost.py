import numpy as np
from gridinfo import C_buy_business, C_buy, expand_array

def EV_earn(mic_EV_load, mic_EV_load_quick, P_basic_dict, EV_Q1, EV_S1, EV_2, EV_3, EV_4):
    total_cost = 0.0
    T_slot = 0.5

    # 将价格向量转换为numpy数组以便于计算
    EV_Q1, EV_S1, EV_2, EV_3, EV_4 = map(np.array, [EV_Q1, EV_S1, EV_2, EV_3, EV_4])
    price_buy_business, price_buy = np.array(expand_array(C_buy_business)), np.array(expand_array(C_buy))

    # 处理P_basic_dict中的基本负荷
    basic_cost = 0.0
    for node, load in P_basic_dict.items():
        load = np.array(load)  # 将负荷转换为numpy数组
        if 101 <= node <= 106:
            # 商业价格
            basic_cost += np.sum(load * price_buy_business * T_slot)
        else:
            # 普通购买价格
            basic_cost += np.sum(load * price_buy * T_slot)

    # 遍历每个微电网
    for grid_id in mic_EV_load:
        # 将负荷字典中的列表转换为numpy数组
        load = np.array(mic_EV_load[grid_id])
        quick_load = np.array(mic_EV_load_quick.get(grid_id, np.zeros(48)))

        # 计算慢充负荷
        slow_load = load - quick_load

        # 选择相应的价格
        if grid_id == 0:
            quick_price = EV_Q1
            slow_price = EV_S1
        elif grid_id == 1:
            slow_price = EV_2
        elif grid_id == 2:
            slow_price = EV_3
        elif grid_id == 3:
            slow_price = EV_4

        # 对于0号微电网，计算快充成本
        if grid_id == 0:
            quick_cost = np.sum(quick_load * quick_price * T_slot)
            total_cost += quick_cost

        # 计算慢充成本
        slow_cost = np.sum(slow_load * slow_price * T_slot)
        total_cost += slow_cost

    # 将基本负荷成本加到总成本中
    total_cost += basic_cost

    return total_cost
