# -*- coding: utf-8 -*-
"""
事件类型平衡权重配置 (ta1-trace-e3-official-1)

权重计算方法: weight = max_count / count
这种方法给出现频率低的事件类型更高的权重，用于处理类别不平衡问题。
"""

# 事件类型平衡权重映射
# 格式: '事件类型名称': 权重值
event_type_balanced_weights = {
    'EVENT_ACCEPT': 3775.397914,  # count=1342
    'EVENT_CHANGE_PRINCIPAL': 3832.514372,  # count=1322
    'EVENT_CLONE': 401.950337,  # count=12605
    'EVENT_CLOSE': 4.828123,  # count=1049390
    'EVENT_CONNECT': 52.288887,  # count=96896
    'EVENT_CREATE_OBJECT': 169.524676,  # count=29887
    'EVENT_EXECUTE': 569.663144,  # count=8894
    'EVENT_EXIT': 274.032344,  # count=18489
    'EVENT_FORK': 524.057096,  # count=9668
    'EVENT_LINK': 21559.931915,  # count=235
    'EVENT_LOADLIBRARY': 258.948380,  # count=19566
    'EVENT_MMAP': 10.493233,  # count=482843
    'EVENT_MODIFY_FILE_ATTRIBUTES': 19947.181102,  # count=254
    'EVENT_MPROTECT': 1.000000,  # count=5066584
    'EVENT_OPEN': 7.465834,  # count=678636
    'EVENT_OTHER': 2533292.000000,  # count=2
    'EVENT_READ': 1.547341,  # count=3274381
    'EVENT_RECVMSG': 1.740408,  # count=2911148
    'EVENT_RENAME': 338.924610,  # count=14949
    'EVENT_SENDMSG': 3.489063,  # count=1452133
    'EVENT_TRUNCATE': 223.680367,  # count=22651
    'EVENT_UNIT': 2.115126,  # count=2395405
    'EVENT_UNLINK': 181.065828,  # count=27982
    'EVENT_UPDATE': 844430.666667,  # count=6
    'EVENT_WRITE': 1.167716,  # count=4338882
}

def get_event_type_weight(event_type_str):
    """
    根据事件类型返回平衡权重
    
    参数:
        event_type_str: 事件类型的字符串表示
    
    返回:
        权重值（float）
    """
    return event_type_balanced_weights.get(event_type_str, 1.0)
