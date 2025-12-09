# -*- coding: utf-8 -*-
"""
节点类型平衡权重配置 (ta1-trace-e3-official-1)

权重计算方法: weight = max_count / count
这种方法给出现频率低的节点类型更高的权重，用于处理类别不平衡问题。
"""

# 节点类型平衡权重映射
# 格式: '节点类型名称': 权重值
node_type_balanced_weights = {
    'FILE_OBJECT_BLOCK': 1340220.000000,  # count=3
    'FILE_OBJECT_CHAR': 30004.925373,  # count=134
    'FILE_OBJECT_DIR': 72.949053,  # count=55116
    'FILE_OBJECT_FILE': 75.187658,  # count=53475
    'FILE_OBJECT_LINK': 17109.191489,  # count=235
    'FILE_OBJECT_UNIX_SOCKET': 191460.000000,  # count=21
    'MemoryObject': 4.671357,  # count=860705
    'NetFlowObject': 17.168001,  # count=234195
    'PRINCIPAL_LOCAL': 138643.448276,  # count=29
    'SRCSINK_UNKNOWN': 1.000000,  # count=4020660
    'SUBJECT_PROCESS': 123.964358,  # count=32434
    'SUBJECT_UNIT': 1.678489,  # count=2395405
    'UnnamedPipeObject': 515.932247,  # count=7793
}

def get_node_type_weight(node_type_str):
    """
    根据节点类型返回平衡权重
    
    参数:
        node_type_str: 节点类型的字符串表示
    
    返回:
        权重值（float）
    """
    return node_type_balanced_weights.get(node_type_str, 1.0)
