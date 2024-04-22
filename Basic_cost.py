import numpy as np
from gridinfo import C_buy_business, C_buy, C_sell, expand_array, end_points

def Basic_earn(p_to_grid, p_from_grid):
    # total_cost = 0.0
    T_slot = 0.5

    price_buy_business, price_buy = np.array(expand_array(C_buy_business)), np.array(expand_array(C_buy))
    price_sell = np.array(expand_array(C_sell))

    node = p_from_grid.columns.tolist()
    string_end_points = [str(col) for col in end_points]  # 将整数转换为字符串
    # 选择在 end_points 中的列，并计算每一列的总和
    power_business = p_from_grid[string_end_points].sum(axis=1).to_numpy()  # 按行求和
    # 剩下的列
    remaining_columns = [col for col in node if col not in string_end_points]
    # 计算剩余列的总和
    power_community = p_from_grid[remaining_columns].sum(axis=1).to_numpy()  # 按行求和

    node = p_to_grid.columns.tolist()
    # 选择在 end_points 中的列，并计算每一列的总和
    re_business = p_to_grid[string_end_points].sum(axis=1).to_numpy()  # 转换为 NumPy 数组
    # 剩下的列
    remaining_columns = [col for col in node if col not in string_end_points]
    # 计算剩余列的总和
    re_community = p_to_grid[remaining_columns].sum(axis=1).to_numpy()  # 转换为 NumPy 数组

    # 计算买电费用
    buy_business_cost = np.sum(power_business * price_buy_business)  # 商业买电费用
    buy_community_cost = np.sum(power_community * price_buy)  # 社区买电费用
    total_buy_cost = buy_business_cost + buy_community_cost  # 总买电费用

    # 计算卖电费用
    sell_business_earn = np.sum(re_business * price_sell)  # 商业卖电费用
    sell_community_earn = np.sum(re_community * price_sell)  # 社区卖电费用
    total_sell_earn = sell_business_earn + sell_community_earn  # 总卖电费用

    # 计算买电与卖电的差值
    total_cost = (total_buy_cost - total_sell_earn) * T_slot  # 买电 - 卖电

    return total_cost
