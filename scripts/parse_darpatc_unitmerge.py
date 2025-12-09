#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_darpatc_unitmerge.py

功能：基于 parse_darpatc.py 创建的新解析器，对数据进行 UnitMerge 操作。

UnitMerge 算法：
1. 遍历所有边
2. 当某条边连接的两个节点名称相等时（且边的类型最好为 EVENT_UNIT）
3. 删除这条边
4. 把出边节点连接的所有边都直连到入边节点
5. 出边节点变成孤立节点，若有必要，可以删除
"""

import time
import os
import os.path as osp
import re
from typing import Dict, List, Set, Tuple, Optional
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
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
# 改进 type 字段的正则表达式，支持更多格式
pattern_type = re.compile(r'\"type\"\s*:\s*\"([^\"]+)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

# 提取节点名称的模式（尝试多种可能的格式）
pattern_name_properties = re.compile(r'\"properties\"\s*:\s*{\s*\"map\"\s*:\s*{\s*\"name\"\s*:\s*\"([^\"]+)\"')
pattern_name1 = re.compile(r'\"name\"\s*:\s*\"([^\"]+)\"')
pattern_name2 = re.compile(r'\"name\"\s*:\s*{\s*\"string\"\s*:\s*\"([^\"]+)\"')
pattern_hostname = re.compile(r'\"hostName\"\s*:\s*\"([^\"]+)\"')

notice_num = 1000000

# 从类名中提取节点类型的正则表达式
pattern_class_name = re.compile(r'com\.bbn\.tc\.schema\.avro\.cdm18\.([A-Za-z0-9_]+)')

def extract_node_type_from_class(line: str) -> Optional[str]:
    """
    从 JSON 行的类名中提取节点类型
    
    :param line: JSON 格式的行
    :return: 节点类型，如果未找到则返回 None
    """
    # 查找所有匹配的类名
    class_matches = pattern_class_name.findall(line)
    if class_matches:
        # 返回第一个匹配的类名（通常是主要的类）
        class_name = class_matches[0]
        # 移除常见的后缀
        if class_name.endswith('Object'):
            return class_name
        elif class_name.startswith('Subject'):
            return class_name
        elif class_name.startswith('Principal'):
            return class_name
        elif class_name.startswith('File'):
            return class_name
        elif class_name.startswith('SrcSink'):
            return class_name
        else:
            return class_name
    return None

def extract_node_name(line: str) -> Optional[str]:
    """
    从 JSON 行中提取节点名称
    
    :param line: JSON 格式的行
    :return: 节点名称，如果未找到则返回 None
    """
    # 优先尝试 properties.map.name 格式（Subject 节点）
    name_matches = pattern_name_properties.findall(line)
    if name_matches:
        return name_matches[0]
    
    # 尝试 hostName 字段（Host 节点）
    name_matches = pattern_hostname.findall(line)
    if name_matches:
        return name_matches[0]
    
    # 尝试直接的 "name":"value" 格式
    name_matches = pattern_name1.findall(line)
    if name_matches:
        # 排除接口名称等（这些通常在 interfaces 数组中）
        if '"interfaces"' not in line or line.find('"name"') < line.find('"interfaces"'):
            return name_matches[0]
    
    # 尝试 "name":{"string":"value"} 格式
    name_matches = pattern_name2.findall(line)
    if name_matches:
        return name_matches[0]
    
    return None

def main():
    # 解压文件（如果需要）
    tar_files = [
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official.json.tar.gz',
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official-2.json.tar.gz',
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-fivedirections-e3-official-2.json.tar.gz',
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-theia-e3-official-1r.json.tar.gz',
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-theia-e3-official-6r.json.tar.gz',
        # '../graphchi-cpp-master/graph_data/darpatc/ta1-trace-e3-official-1.json.tar.gz',
    ]
    
    for tar_file in tar_files:
        json_file = tar_file.replace('.tar.gz', '').split('/')[-1]
        if not osp.exists(json_file):
            if osp.exists(tar_file):
                show(f"解压文件: {tar_file}")
                os.system(f'tar -zxvf {tar_file}')
    
    # 数据集列表
    path_list = [
        # 'ta1-cadets-e3-official.json',
        # 'ta1-cadets-e3-official-2.json',
        # 'ta1-theia-e3-official-1r.json',
        # 'ta1-theia-e3-official-6r.json',
        'ta1-trace-e3-official-1.json'
    ]
    
    for path in path_list:
        if not osp.exists(path):
            show(f"跳过不存在的文件: {path}")
            continue
            
        show(f"开始处理: {path}")
        
        # 第一遍：构建节点ID到节点类型和节点名称的映射
        id_nodetype_map: Dict[str, str] = {}
        id_nodename_map: Dict[str, str] = {}
        
        show("第一遍：提取节点信息...")
        for i in range(100):
            now_path = path + '.' + str(i)
            if i == 0:
                now_path = path
            if not osp.exists(now_path):
                break
            
            f = open(now_path, 'r', encoding='utf-8')
            show(f"读取节点信息: {now_path}")
            cnt = 0
            
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    safe_print(f"  已处理 {cnt} 行")
                
                # 跳过 Event 和 Host 类型（这些是边，不是节点）
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                    continue
                
                if len(pattern_uuid.findall(line)) == 0:
                    continue
                
                uuid = pattern_uuid.findall(line)[0]
                subject_type = pattern_type.findall(line)
                
                # 提取节点类型
                node_type = None
                if len(subject_type) >= 1:
                    # 优先使用 type 字段
                    node_type = subject_type[0]
                else:
                    # 如果没有 type 字段，尝试从类名中提取
                    # 先检查常见的特殊类型
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        node_type = 'MemoryObject'
                    elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        node_type = 'NetFlowObject'
                    elif 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        node_type = 'UnnamedPipeObject'
                    elif 'com.bbn.tc.schema.avro.cdm18.FileObject' in line:
                        # FileObject 可能有子类型，需要进一步判断
                        # 注意：FileObject 的子类型通常在 type 字段中，如果没有 type 字段，使用默认值
                        node_type = 'FileObject'  # 默认类型
                    elif 'com.bbn.tc.schema.avro.cdm18.Subject' in line:
                        # Subject 可能有子类型
                        node_type = 'Subject'  # 默认类型
                    elif 'com.bbn.tc.schema.avro.cdm18.Principal' in line:
                        # Principal 可能有子类型
                        node_type = 'Principal'  # 默认类型
                    elif 'com.bbn.tc.schema.avro.cdm18.SrcSink' in line:
                        node_type = 'SRCSINK_UNKNOWN'
                    else:
                        # 尝试从类名中提取
                        extracted_type = extract_node_type_from_class(line)
                        if extracted_type:
                            node_type = extracted_type
                
                # 如果找到了节点类型，保存它
                if node_type:
                    id_nodetype_map[uuid] = node_type
                else:
                    # 如果仍然找不到，尝试从类名中提取（作为最后的手段）
                    extracted_type = extract_node_type_from_class(line)
                    if extracted_type:
                        id_nodetype_map[uuid] = extracted_type
                    # 如果还是找不到，记录警告但继续处理（不添加到映射中，后续边处理时会跳过）
                    # 注意：这里不添加节点到映射中，意味着后续处理边时如果遇到这个节点ID会被跳过
                
                # 提取节点名称
                node_name = extract_node_name(line)
                if node_name:
                    id_nodename_map[uuid] = node_name
            
            f.close()
        
        show(f"共提取了 {len(id_nodetype_map)} 个节点的类型")
        show(f"共提取了 {len(id_nodename_map)} 个节点的名称")
        
        # 第二遍：提取所有边，并记录每条边来自哪个分片
        show("第二遍：提取所有边...")
        all_edges: List[Tuple[str, str, str, str, str, str, int]] = []  # (srcId, srcType, dstId, dstType, edgeType, timestamp, fragment_id)
        
        fragment_list = []
        for i in range(100):
            now_path = path + '.' + str(i)
            if i == 0:
                now_path = path
            if not osp.exists(now_path):
                break
            
            fragment_list.append(i)
            f = open(now_path, 'r', encoding='utf-8')
            show(f"处理边信息: {now_path}")
            cnt = 0
            
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    safe_print(f"  已处理 {cnt} 行")
                
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    edgeType = pattern_type.findall(line)
                    if len(edgeType) == 0:
                        continue
                    edgeType = edgeType[0]
                    
                    timestamp = pattern_time.findall(line)
                    if len(timestamp) == 0:
                        continue
                    timestamp = timestamp[0]
                    
                    srcId = pattern_src.findall(line)
                    if len(srcId) == 0:
                        continue
                    srcId = srcId[0]
                    
                    if srcId not in id_nodetype_map:
                        continue
                    srcType = id_nodetype_map[srcId]
                    
                    # 处理第一个目标节点
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if dstId1 in id_nodetype_map:
                            dstType1 = id_nodetype_map[dstId1]
                            all_edges.append((srcId, srcType, dstId1, dstType1, edgeType, timestamp, i))
                    
                    # 处理第二个目标节点
                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if dstId2 in id_nodetype_map:
                            dstType2 = id_nodetype_map[dstId2]
                            all_edges.append((srcId, srcType, dstId2, dstType2, edgeType, timestamp, i))
            
            f.close()
        
        show(f"共提取了 {len(all_edges)} 条边")
        
        # 第三遍：执行 UnitMerge 操作
        show("第三遍：执行 UnitMerge 操作...")
        
        # 找到需要合并的边：两个节点名称相等，且边类型为 EVENT_UNIT
        edges_to_remove: Set[int] = set()  # 需要删除的边的索引
        merge_map: Dict[str, str] = {}  # 出边节点ID -> 入边节点ID 的映射
        
        for idx, (srcId, srcType, dstId, dstType, edgeType, timestamp, fragment_id) in enumerate(all_edges):
            # 只处理 EVENT_UNIT 类型的边
            if edgeType != 'EVENT_UNIT':
                continue
            
            # 获取节点名称
            srcName = id_nodename_map.get(srcId)
            dstName = id_nodename_map.get(dstId)
            
            # 如果两个节点名称相等且都不为空
            if srcName and dstName and srcName == dstName:
                # 标记这条边需要删除
                edges_to_remove.add(idx)
                # 记录合并映射：将出边节点（dst）合并到入边节点（src）
                # 注意：如果 dstId 已经在 merge_map 中，需要找到最终的合并目标
                final_target = srcId
                while final_target in merge_map:
                    final_target = merge_map[final_target]
                if dstId != final_target:
                    merge_map[dstId] = final_target
                    show(f"合并节点: {dstId} ({dstName}) -> {final_target} ({id_nodename_map.get(final_target, 'unknown')})")
        
        show(f"找到 {len(edges_to_remove)} 条需要删除的边")
        show(f"需要合并 {len(merge_map)} 个节点")
        
        # 应用合并映射：将所有指向出边节点的边改为指向入边节点
        # 将所有从出边节点出发的边改为从入边节点出发
        merged_edges: List[Tuple[str, str, str, str, str, str, int]] = []
        
        for idx, (srcId, srcType, dstId, dstType, edgeType, timestamp, fragment_id) in enumerate(all_edges):
            # 跳过需要删除的边
            if idx in edges_to_remove:
                continue
            
            # 应用合并映射（找到最终的合并目标）
            original_srcId = srcId
            original_dstId = dstId
            
            # 如果源节点需要合并，找到最终的合并目标
            while srcId in merge_map:
                srcId = merge_map[srcId]
                srcType = id_nodetype_map.get(srcId, srcType)
            
            # 如果目标节点需要合并，找到最终的合并目标
            while dstId in merge_map:
                dstId = merge_map[dstId]
                dstType = id_nodetype_map.get(dstId, dstType)
            
            # 避免自环（如果合并后源节点和目标节点相同，跳过这条边）
            if srcId == dstId:
                continue
            
            merged_edges.append((srcId, srcType, dstId, dstType, edgeType, timestamp, fragment_id))
        
        show(f"合并后剩余 {len(merged_edges)} 条边（删除了 {len(edges_to_remove)} 条边）")
        
        # 第四遍：写入输出文件（按原始分片分配）
        show("第四遍：写入输出文件...")
        
        # 按分片组织边
        fragment_edges: Dict[int, List[Tuple[str, str, str, str, str, str]]] = defaultdict(list)
        for srcId, srcType, dstId, dstType, edgeType, timestamp, fragment_id in merged_edges:
            fragment_edges[fragment_id].append((srcId, srcType, dstId, dstType, edgeType, timestamp))
        
        for i in fragment_list:
            now_path = path + '.' + str(i)
            if i == 0:
                now_path = path
            
            output_path = now_path + '.unitmerge.txt'
            fw = open(output_path, 'w', encoding='utf-8')
            
            # 写入该分片对应的边
            for srcId, srcType, dstId, dstType, edgeType, timestamp in fragment_edges[i]:
                this_edge = f"{srcId}\t{srcType}\t{dstId}\t{dstType}\t{edgeType}\t{timestamp}\n"
                fw.write(this_edge)
            
            fw.close()
            show(f"已写入分片 {i}: {len(fragment_edges[i])} 条边")
        
        show(f"完成处理: {path}")
        show(f"统计信息:")
        show(f"  - 原始边数量: {len(all_edges)}")
        show(f"  - 删除边数量: {len(edges_to_remove)}")
        show(f"  - 合并节点数量: {len(merge_map)}")
        show(f"  - 最终边数量: {len(merged_edges)}")
    
    # 复制文件到目标目录（根据原始脚本的逻辑）
    show("复制文件到目标目录...")
    os.system('cp ta1-theia-e3-official-1r.json.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/theia_train_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-theia-e3-official-6r.json.8.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/theia_test_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-cadets-e3-official.json.1.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_train_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-cadets-e3-official-2.json.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_test_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-fivedirections-e3-official-2.json.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_train_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-fivedirections-e3-official-2.json.23.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_test_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-trace-e3-official-1.json.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/trace_train_unitmerge.txt 2>/dev/null || true')
    os.system('cp ta1-trace-e3-official-1.json.4.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/trace_test_unitmerge.txt 2>/dev/null || true')
    
    show("所有处理完成！")

if __name__ == '__main__':
    main()

