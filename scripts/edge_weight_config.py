# -*- coding: utf-8 -*-
"""
边类型权重配置文件
可以根据不同边类型的重要性设置不同的权重值

使用方法：
1. 在weight_map字典中添加边类型和对应的权重
2. 权重值可以是任意正数，建议范围在0.1到10.0之间
3. 默认权重为1.0（如果边类型不在映射中）
"""

# 边类型权重映射
# 格式: '边类型名称': 权重值
weight_map = {
	# 示例：可以根据实际DARPA数据集中的边类型添加权重
	# 'EVENT_READ': 1.5,
	# 'EVENT_WRITE': 1.5,
	# 'EVENT_EXECUTE': 2.0,
	# 'EVENT_OPEN': 1.0,
	# 'EVENT_CLOSE': 1.0,
	# 'EVENT_CONNECT': 2.5,
	# 'EVENT_ACCEPT': 2.5,
	# 'EVENT_SEND': 1.8,
	# 'EVENT_RECV': 1.8,
	# 可以根据需要添加更多边类型
}

# 默认权重（当边类型不在weight_map中时使用）
DEFAULT_WEIGHT = 1.0

def get_edge_weight(edge_type_str):
	"""
	根据边类型返回权重
	
	参数:
		edge_type_str: 边类型的字符串表示
	
	返回:
		权重值（float）
	"""
	return weight_map.get(edge_type_str, DEFAULT_WEIGHT)

