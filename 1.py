import pandapower as pp

# 创建一个空的pandapower网络
net = pp.create_empty_network()

# 列出所有线路的标准类型
std_types = pp.available_std_types(net, element='line')
print("Available standard types for lines:\n", std_types)

# 假设你对某个特定的标准类型感兴趣，比如 "NAYY 4x50 SE"（这只是一个例子，可能并不在标准库中）
# 你可以查看这个类型的参数，如果它在标准类型库中的话
# std_type_name = "NAYY 4x50 SE"  # 替换成实际可用的标准类型名称
# if std_type_name in std_types['line'].keys():
#     print("Parameters for", std_type_name, ":\n", pp.get_std_type(net, std_type_name, element='line'))
# else:
#     print(std_type_name, "is not an available standard type.")
