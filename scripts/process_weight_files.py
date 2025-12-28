# -*- coding: utf-8 -*-
"""
处理weight文件夹下的文件，计算事件类型的加权权重并归一化

功能：
1. 读取weight文件夹下的trace_train, theia_train, cadets_train文件夹内的所有文件
2. 从每个文件中提取事件序列和JSON回答
3. 计算每个文件对应的权重（三个分数的平均值）
4. 统计每个事件类型的加权和
5. 使用公式 final = 0.5 + (max - cur) / (max - min) 归一化
"""

import os
import os.path as osp
import json
import re
import time
from collections import defaultdict

def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def parse_weight_file(file_path):
    """
    解析权重文件，提取事件序列和JSON回答
    
    返回:
        events: 事件列表，每个事件是 (subject, event_type, object) 的元组
        weight: 权重值（三个分数的平均值）
    """
    events = []
    weight = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找JSON回答部分
    json_match = re.search(r'--- 回答 ---\s*\n\s*(\{.*?\})', content, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            answer = json.loads(json_str)
            temporal_score = answer.get('temporal_score', 0)
            contextual_score = answer.get('contextual_score', 0)
            propagational_score = answer.get('propagational_score', 0)
            # 计算平均值作为权重
            weight = (temporal_score + contextual_score + propagational_score) / 3.0
        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告: 无法解析JSON回答 in {file_path}: {e}")
            return events, None
    
    # 提取事件序列（格式：subject,event_type,object）
    # 跳过问题描述行（包含"====="或"You will be provided"等）
    lines = content.split('\n')
    in_events_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过问题标题和描述
        if line.startswith('=====') or 'You will be provided' in line:
            in_events_section = True
            continue
        
        # 如果遇到"--- 回答 ---"，停止解析事件
        if line.startswith('--- 回答 ---'):
            break
        
        # 解析事件行（格式：subject,event_type,object）
        if in_events_section and ',' in line:
            parts = line.split(',')
            if len(parts) >= 3:
                subject = parts[0].strip()
                event_type = parts[1].strip()
                obj = ','.join(parts[2:]).strip()  # 处理object中可能包含逗号的情况
                if event_type.startswith('EVENT_'):
                    events.append((subject, event_type, obj))
    
    return events, weight

def process_dataset(dataset_name, weight_dir):
    """
    处理单个数据集的所有文件
    
    参数:
        dataset_name: 数据集名称（trace_train, theia_train, cadets_train）
        weight_dir: weight文件夹路径
    
    返回:
        event_weights: 字典，{event_type: weighted_sum}
    """
    dataset_path = osp.join(weight_dir, dataset_name)
    
    if not osp.exists(dataset_path):
        print(f"警告: 数据集路径不存在: {dataset_path}")
        return {}
    
    # 统计每个事件类型的加权和
    event_weighted_sums = defaultdict(float)
    
    # 获取所有文件
    files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
    total_files = len(files)
    
    show(f"开始处理数据集 {dataset_name}，共 {total_files} 个文件")
    
    processed = 0
    skipped = 0
    
    for filename in files:
        file_path = osp.join(dataset_path, filename)
        events, weight = parse_weight_file(file_path)
        
        if weight is None:
            skipped += 1
            continue
        
        # 对该文件中的所有事件，累加权重
        for subject, event_type, obj in events:
            event_weighted_sums[event_type] += weight
        
        processed += 1
        
        if processed % 100 == 0:
            print(f"  已处理 {processed}/{total_files} 个文件")
    
    show(f"数据集 {dataset_name} 处理完成: 成功 {processed} 个，跳过 {skipped} 个")
    
    return dict(event_weighted_sums)

def normalize_weights(event_weights):
    """
    使用公式 final = 0.5 + (max - cur) / (max - min) 归一化权重
    
    参数:
        event_weights: 字典，{event_type: weighted_sum}
    
    返回:
        normalized_weights: 字典，{event_type: normalized_weight}
    """
    if not event_weights:
        return {}
    
    values = list(event_weights.values())
    max_val = max(values)
    min_val = min(values)
    
    # 如果最大值和最小值相等，所有值归一化为0.5
    if max_val == min_val:
        return {event_type: 0.5 for event_type in event_weights.keys()}
    
    normalized_weights = {}
    for event_type, weighted_sum in event_weights.items():
        normalized = 0.5 + (max_val - weighted_sum) / (max_val - min_val)
        normalized_weights[event_type] = normalized
    
    return normalized_weights

def main():
    # weight文件夹路径
    weight_dir = osp.join(osp.dirname(__file__), 'weight')
    
    # 要处理的数据集
    datasets = ['trace_train', 'theia_train', 'cadets_train']
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_name in datasets:
        show(f"\n{'='*60}")
        show(f"处理数据集: {dataset_name}")
        show(f"{'='*60}")
        
        event_weights = process_dataset(dataset_name, weight_dir)
        
        if event_weights:
            normalized_weights = normalize_weights(event_weights)
            
            all_results[dataset_name] = {
                'raw_weights': event_weights,
                'normalized_weights': normalized_weights,
                'statistics': {
                    'total_events': sum(event_weights.values()),
                    'unique_event_types': len(event_weights),
                    'max_weight': max(event_weights.values()),
                    'min_weight': min(event_weights.values()),
                }
            }
            
            # 输出结果摘要
            show(f"\n数据集 {dataset_name} 结果摘要:")
            show(f"  事件类型数量: {len(event_weights)}")
            show(f"  总加权和: {sum(event_weights.values()):.2f}")
            show(f"  最大加权和: {max(event_weights.values()):.2f}")
            show(f"  最小加权和: {min(event_weights.values()):.2f}")
    
    # 保存结果到JSON文件
    output_file = osp.join(osp.dirname(__file__), 'weight', 'event_weights_result.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    show(f"\n结果已保存到: {output_file}")
    
    # 生成Python配置文件
    for dataset_name in datasets:
        if dataset_name not in all_results:
            continue
        
        normalized_weights = all_results[dataset_name]['normalized_weights']
        
        # 生成Python配置文件
        config_file = osp.join(osp.dirname(__file__), f'edge_weight_config_{dataset_name}.py')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n')
            f.write(f'"""\n')
            f.write(f'边类型权重配置文件 - {dataset_name}\n')
            f.write(f'从weight文件计算得到的归一化权重\n')
            f.write(f'归一化公式: final = 0.5 + (max - cur) / (max - min)\n')
            f.write(f'"""\n\n')
            f.write('# 权重开关：设置为 False 可以完全关闭权重功能\n')
            f.write('ENABLE_WEIGHT = True\n\n')
            f.write('# 边类型权重映射\n')
            f.write('# 格式: \'边类型名称\': 权重值\n')
            f.write('weight_map = {\n')
            
            # 按权重值排序输出
            sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
            for event_type, weight in sorted_weights:
                f.write(f"    '{event_type}': {weight:.6f},\n")
            
            f.write('}\n\n')
            f.write('# 默认权重（当边类型不在weight_map中时使用）\n')
            f.write('DEFAULT_WEIGHT = 1.0\n\n')
            f.write('def get_edge_weight(edge_type_str):\n')
            f.write('    """\n')
            f.write('    根据边类型返回权重\n')
            f.write('    \n')
            f.write('    参数:\n')
            f.write('        edge_type_str: 边类型的字符串表示\n')
            f.write('    \n')
            f.write('    返回:\n')
            f.write('        权重值（float）\n')
            f.write('    """\n')
            f.write('    return weight_map.get(edge_type_str, DEFAULT_WEIGHT)\n')
        
        show(f"配置文件已生成: {config_file}")
    
    show("\n处理完成！")

if __name__ == '__main__':
    main()

