# -*- coding: utf-8 -*-
"""
边类型权重配置文件 - theia_train
从weight文件计算得到的归一化权重
归一化公式: final = 0.5 + (max - cur) / (max - min)
"""

# 权重开关：设置为 False 可以完全关闭权重功能
ENABLE_WEIGHT = True

# 边类型权重映射
# 格式: '边类型名称': 权重值
weight_map = {
    'EVENT_MODIFY_FILE_ATTRIBUTES': 1.500000,
    'EVENT_SHM': 1.498911,
    'EVENT_WRITE_SOCKET_PARAMS': 1.498192,
    'EVENT_BOOT': 1.360142,
    'EVENT_WRITE': 1.157760,
    'EVENT_CLONE': 1.152127,
    'EVENT_UNLINK': 1.150177,
    'EVENT_READ_SOCKET_PARAMS': 1.075078,
    'EVENT_EXECUTE': 1.068407,
    'EVENT_SENDMSG': 1.065993,
    'EVENT_RECVMSG': 1.000053,
    'EVENT_SENDTO': 0.968052,
    'EVENT_RECVFROM': 0.952664,
    'EVENT_MPROTECT': 0.899170,
    'EVENT_CONNECT': 0.889639,
    'EVENT_MMAP': 0.875939,
    'EVENT_READ': 0.516760,
    'EVENT_OPEN': 0.500000,
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
