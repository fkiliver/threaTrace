# -*- coding: utf-8 -*-
"""
边类型权重配置文件 - cadets_train
从weight文件计算得到的归一化权重
归一化公式: final = 0.5 + (max - cur) / (max - min)
"""

# 权重开关：设置为 False 可以完全关闭权重功能
ENABLE_WEIGHT = True

# 边类型权重映射
# 格式: '边类型名称': 权重值
weight_map = {
    'EVENT_RECVFROM': 1.500000,
    'EVENT_RENAME': 1.499605,
    'EVENT_MODIFY_FILE_ATTRIBUTES': 1.499569,
    'EVENT_LINK': 1.499569,
    'EVENT_ACCEPT': 1.499566,
    'EVENT_UNLINK': 1.499125,
    'EVENT_SENDTO': 1.498291,
    'EVENT_CONNECT': 1.496570,
    'EVENT_LSEEK': 1.496081,
    'EVENT_WRITE': 1.492692,
    'EVENT_EXECUTE': 1.468893,
    'EVENT_MMAP': 1.464063,
    'EVENT_CHANGE_PRINCIPAL': 1.437436,
    'EVENT_FORK': 1.430932,
    'EVENT_CREATE_OBJECT': 1.402262,
    'EVENT_MODIFY_PROCESS': 1.194070,
    'EVENT_EXIT': 1.176360,
    'EVENT_READ': 0.686914,
    'EVENT_OPEN': 0.583848,
    'EVENT_CLOSE': 0.500000,
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
