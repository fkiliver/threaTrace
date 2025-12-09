#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stat_node_type_frequency_trace.py

功能：统计 ta1-trace-e3-official-1 各个分片中各个类型的事件（Event）出现频率，
     并使用这个频率作为 balanced weights。

权重计算方法: weight = max_count / count
这种方法给出现频率低的事件类型更高的权重，用于处理类别不平衡问题。
"""

import time
import os
import os.path as osp
import re
from typing import Dict, List, Tuple
from collections import defaultdict

def safe_print(*args, **kwargs):
    """
    安全打印函数，避免 UnicodeEncodeError
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        ascii_args = []
        for a in args:
            if isinstance(a, str):
                ascii_args.append(a.encode("ascii", errors="ignore").decode("ascii", errors="ignore"))
            else:
                ascii_args.append(a)
        print(*ascii_args, **kwargs)

def show(str):
    """带时间戳的打印函数"""
    safe_print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

# 正则表达式模式
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_type = re.compile(r'type\":\"(.*?)\"')

notice_num = 1000000

def main():
    # 输入文件路径
    input_path = 'ta1-trace-e3-official-1.json'
    
    # 输出文件
    output_stats_file = 'ta1-trace-e3-official-1_event_type_statistics.txt'
    output_weights_file = 'ta1-trace-e3-official-1_event_type_balanced_weights.py'
    
    # 检查文件是否存在，如果不存在则尝试解压
    if not osp.exists(input_path):
        show(f"文件 {input_path} 不存在，尝试解压...")
        tar_file = '../graphchi-cpp-master/graph_data/darpatc/ta1-trace-e3-official-1.json.tar.gz'
        if osp.exists(tar_file):
            show(f"正在解压: {tar_file}")
            os.system(f'tar -zxvf {tar_file}')
        else:
            show(f"错误: 找不到文件 {input_path} 或压缩包 {tar_file}")
            return
    
    show(f"开始处理: {input_path}")
    
    # 统计每个分片中各个事件类型的出现频率
    # fragment_stats[fragment_id][event_type] = count
    fragment_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    # 全局统计（所有分片合并）
    global_stats: Dict[str, int] = defaultdict(int)
    
    # 第一遍：遍历所有分片文件，统计事件类型
    fragment_list = []
    for i in range(100):
        now_path = input_path + '.' + str(i)
        if i == 0:
            now_path = input_path
        if not osp.exists(now_path):
            break
        
        fragment_list.append(i)
        show(f"处理分片 {i}: {now_path}")
        
        f = open(now_path, 'r', encoding='utf-8')
        cnt = 0
        
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                safe_print(f"  已处理 {cnt} 行")
            
            # 只处理 Event 类型（事件）
            if 'com.bbn.tc.schema.avro.cdm18.Event' not in line:
                continue
            
            # 提取事件类型
            event_types = pattern_type.findall(line)
            if len(event_types) == 0:
                continue
            
            event_type = event_types[0]
            
            if event_type:
                fragment_stats[i][event_type] += 1
                global_stats[event_type] += 1
        
        f.close()
        show(f"分片 {i} 完成，共统计到 {sum(fragment_stats[i].values())} 个事件")
    
    show(f"共处理 {len(fragment_list)} 个分片文件")
    show(f"共统计到 {len(global_stats)} 种不同的事件类型")
    
    # 计算 balanced weights
    if len(global_stats) == 0:
        show("错误: 未找到任何事件类型，无法计算权重")
        return
    
    max_count = max(global_stats.values())
    event_type_balanced_weights = {}
    for event_type, count in sorted(global_stats.items()):
        weight = max_count / count if count > 0 else 1.0
        event_type_balanced_weights[event_type] = weight
    
    # 写入统计结果文件
    show(f"写入统计结果到: {output_stats_file}")
    with open(output_stats_file, 'w', encoding='utf-8') as fw:
        fw.write(f"# ta1-trace-e3-official-1 事件类型统计结果\n")
        fw.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        fw.write(f"# 共处理 {len(fragment_list)} 个分片文件\n\n")
        
        # 写入全局统计
        fw.write("=" * 80 + "\n")
        fw.write("全局统计（所有分片合并）\n")
        fw.write("=" * 80 + "\n")
        fw.write(f"{'事件类型':<40} {'出现次数':<15} {'权重':<15}\n")
        fw.write("-" * 80 + "\n")
        for event_type, count in sorted(global_stats.items(), key=lambda x: x[1], reverse=True):
            weight = event_type_balanced_weights[event_type]
            fw.write(f"{event_type:<40} {count:<15} {weight:<15.6f}\n")
        fw.write("\n")
        
        # 写入各分片统计
        fw.write("=" * 80 + "\n")
        fw.write("各分片统计详情\n")
        fw.write("=" * 80 + "\n")
        for fragment_id in sorted(fragment_list):
            fw.write(f"\n分片 {fragment_id}:\n")
            fw.write(f"{'事件类型':<40} {'出现次数':<15}\n")
            fw.write("-" * 80 + "\n")
            for event_type, count in sorted(fragment_stats[fragment_id].items(), key=lambda x: x[1], reverse=True):
                fw.write(f"{event_type:<40} {count:<15}\n")
            fw.write(f"分片 {fragment_id} 总计: {sum(fragment_stats[fragment_id].values())} 个事件\n")
    
    # 写入权重配置文件
    show(f"写入权重配置到: {output_weights_file}")
    with open(output_weights_file, 'w', encoding='utf-8') as fw:
        fw.write("# -*- coding: utf-8 -*-\n")
        fw.write('"""\n')
        fw.write("事件类型平衡权重配置 (ta1-trace-e3-official-1)\n")
        fw.write("\n")
        fw.write("权重计算方法: weight = max_count / count\n")
        fw.write("这种方法给出现频率低的事件类型更高的权重，用于处理类别不平衡问题。\n")
        fw.write('"""\n\n')
        fw.write("# 事件类型平衡权重映射\n")
        fw.write("# 格式: '事件类型名称': 权重值\n")
        fw.write("event_type_balanced_weights = {\n")
        
        # 按事件类型名称排序
        for event_type in sorted(event_type_balanced_weights.keys()):
            weight = event_type_balanced_weights[event_type]
            count = global_stats[event_type]
            fw.write(f"    '{event_type}': {weight:.6f},  # count={count}\n")
        
        fw.write("}\n\n")
        fw.write("def get_event_type_weight(event_type_str):\n")
        fw.write("    \"\"\"\n")
        fw.write("    根据事件类型返回平衡权重\n")
        fw.write("    \n")
        fw.write("    参数:\n")
        fw.write("        event_type_str: 事件类型的字符串表示\n")
        fw.write("    \n")
        fw.write("    返回:\n")
        fw.write("        权重值（float）\n")
        fw.write("    \"\"\"\n")
        fw.write("    return event_type_balanced_weights.get(event_type_str, 1.0)\n")
    
    # 打印摘要信息
    show("=" * 80)
    show("统计摘要:")
    show(f"  - 处理的分片数量: {len(fragment_list)}")
    show(f"  - 事件类型总数: {len(global_stats)}")
    show(f"  - 事件总数: {sum(global_stats.values())}")
    show(f"  - 最大出现次数: {max_count}")
    show(f"  - 最小出现次数: {min(global_stats.values())}")
    show("=" * 80)
    show("事件类型统计（按出现次数降序）:")
    for event_type, count in sorted(global_stats.items(), key=lambda x: x[1], reverse=True):
        weight = event_type_balanced_weights[event_type]
        show(f"  {event_type:<40} count={count:<10} weight={weight:.6f}")
    show("=" * 80)
    show(f"完成！结果已保存到:")
    show(f"  - 统计结果: {output_stats_file}")
    show(f"  - 权重配置: {output_weights_file}")

if __name__ == '__main__':
    main()

