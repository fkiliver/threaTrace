# -*- coding: utf-8 -*-
"""
边类型权重配置文件 - trace_train
从weight文件计算得到的归一化权重
归一化公式: final = 0.5 + (max - cur) / (max - min)
"""

# 权重开关：设置为 False 可以完全关闭权重功能
ENABLE_WEIGHT = True

# 边类型权重映射
# 格式: '边类型名称': 权重值
weight_map = {
    'EVENT_RENAME': 1.500000,
    'EVENT_CREATE_OBJECT': 1.499565,
    'EVENT_TRUNCATE': 1.499173,
    'EVENT_UNLINK': 1.498780,
    'EVENT_CONNECT': 1.494376,
    'EVENT_EXECUTE': 1.465611,
    'EVENT_FORK': 1.459001,
    'EVENT_MPROTECT': 1.458664,
    'EVENT_LOADLIBRARY': 1.443619,
    'EVENT_SENDMSG': 1.345851,
    'EVENT_RECVMSG': 1.124358,
    'EVENT_CLONE': 1.049677,
    'EVENT_EXIT': 0.922307,
    'EVENT_UNIT': 0.887758,
    'EVENT_MMAP': 0.876927,
    'EVENT_READ': 0.856841,
    'EVENT_OPEN': 0.707267,
    'EVENT_CLOSE': 0.556021,
    'EVENT_WRITE': 0.500000,
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
