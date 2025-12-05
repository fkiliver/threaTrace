#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_darpatc_unitmerge.py

功能：基于 parse_darpatc.py 创建的新解析器，实现 UnitMerge 算法。

UnitMerge 算法：
1. 遍历所有边，当某条边连接的两个节点名称相等时（且边的类型最好为unit时）
2. 删除这条边
3. 把出边节点连接的所有边都直连到入边节点
4. 出边节点变成孤立节点，若有必要可以删除

输入：JSON 格式的 DARPA TC 数据文件
输出：经过 UnitMerge 处理后的边列表文件
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
    在某些默认编码为 ASCII 的环境中安全打印（避免 UnicodeEncodeError）。
    如果遇到编码错误，会自动去掉非 ASCII 字符后再打印。
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
        # 再次打印已经转换为 ASCII 的内容
        print(*ascii_args, **kwargs)

def show(str):
    """
    带时间戳的打印函数
    """
    safe_print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

# 解压文件（如果需要）
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official.json.tar.gz')
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official-2.json.tar.gz')
os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-trace-e3-official-1.json.tar.gz')

# path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json']
path_list = ['ta1-trace-e3-official-1.json']
# path_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']
# path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json', 'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']

# 正则表达式模式
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
# 提取节点名称的模式（尝试多种可能的格式）
# 注意：JSON 中名称字段可能有多种格式，需要尝试多种模式
# 1. properties.map.name 格式（最常见，用于 Subject 节点）
pattern_name_properties = re.compile(r'\"properties\"\s*:\s*{\s*\"map\"\s*:\s*{\s*\"name\"\s*:\s*\"([^\"]+)\"')
# 2. 直接的 "name":"value" 格式
pattern_name1 = re.compile(r'\"name\"\s*:\s*\"([^\"]+)\"')  # "name":"value"
# 3. "name":{"string":"value"} 格式
pattern_name2 = re.compile(r'\"name\"\s*:\s*{\s*\"string\"\s*:\s*\"([^\"]+)\"')  # "name":{"string":"value"}
# 4. hostName 字段（用于 Host 节点）
pattern_hostname = re.compile(r'\"hostName\"\s*:\s*\"([^\"]+)\"')

notice_num = 1000000

def extract_node_name(line: str) -> Optional[str]:
    """
    从 JSON 行中提取节点名称
    
    根据 ta1-trace-e3-official-1.json 的格式：
    - Subject 节点的名称在 properties.map.name 字段中
    - Host 节点的名称在 hostName 字段中
    - 其他节点可能有不同的格式
    
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

def unitmerge_process(edges: List[Tuple[str, str, str, str, str, str]], 
                      node_name_map: Dict[str, str],
                      node_type_map: Dict[str, str],
                      unit_edge_types: Set[str] = None) -> List[Tuple[str, str, str, str, str, str]]:
    """
    执行 UnitMerge 算法
    
    :param edges: 边列表，每个元素为 (srcId, srcType, dstId, dstType, edgeType, timestamp)
    :param node_name_map: 节点ID到节点名称的映射
    :param node_type_map: 节点ID到节点类型的映射
    :param unit_edge_types: 被认为是 unit 类型的边类型集合（如果为None，则包含所有包含"unit"的边类型）
    :return: 处理后的边列表
    """
    if unit_edge_types is None:
        # 默认包含所有包含"unit"的边类型（不区分大小写）
        all_edge_types = set([e[4] for e in edges])
        unit_edge_types = {et for et in all_edge_types if 'unit' in et.lower()}
        if len(unit_edge_types) == 0:
            show("警告: 未找到包含'unit'的边类型，将检查所有边类型")
            unit_edge_types = all_edge_types  # 如果没有找到unit类型，检查所有边
    
    # 构建图结构：记录每个节点的出边和入边
    # out_edges[node_id] = [(dst_id, edge_type, timestamp, edge_index), ...]
    # in_edges[node_id] = [(src_id, edge_type, timestamp, edge_index), ...]
    out_edges: Dict[str, List[Tuple[str, str, str, int]]] = defaultdict(list)
    in_edges: Dict[str, List[Tuple[str, str, str, int]]] = defaultdict(list)
    
    # 记录需要删除的边索引
    edges_to_remove: Set[int] = set()
    # 按(源节点名称, 目标节点名称)分组，记录需要合并的边组
    # edge_groups[(src_name, dst_name)] = [(src_id, dst_id, edge_idx), ...]
    edge_groups: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = defaultdict(list)
    
    # 第一遍遍历：构建图结构并找出需要合并的边
    for idx, (src_id, src_type, dst_id, dst_type, edge_type, timestamp) in enumerate(edges):
        # 构建图结构
        out_edges[src_id].append((dst_id, edge_type, timestamp, idx))
        in_edges[dst_id].append((src_id, edge_type, timestamp, idx))
        
        # 检查是否是 unit 类型的边
        is_unit_edge = edge_type in unit_edge_types
        
        # 获取节点名称
        src_name = node_name_map.get(src_id, "")
        dst_name = node_name_map.get(dst_id, "")
        
        # 如果源节点名称和目标节点名称都存在，且是 unit 类型的边，按名称分组
        if src_name and dst_name and is_unit_edge:
            edge_groups[(src_name, dst_name)].append((src_id, dst_id, idx))
    
    # 找出需要合并的节点对：对于每个(源节点名称, 目标节点名称)组，
    # 如果有多条边连接不同的节点ID，则需要合并
    nodes_to_merge: List[Tuple[str, str, List[Tuple[str, str, int]]]] = []  # (代表src_id, 代表dst_id, 所有边列表)
    
    for (src_name, dst_name), edge_list in edge_groups.items():
        # 如果这个名称对只有一条边，不需要合并
        if len(edge_list) <= 1:
            continue
        
        # 检查是否有不同的节点ID对
        unique_node_pairs = set((src_id, dst_id) for src_id, dst_id, _ in edge_list)
        if len(unique_node_pairs) > 1:
            # 选择第一个节点对作为代表
            representative_src_id, representative_dst_id, _ = edge_list[0]
            nodes_to_merge.append((representative_src_id, representative_dst_id, edge_list))
            
            # 标记需要删除的边（保留第一条，删除其他的）
            for src_id, dst_id, edge_idx in edge_list[1:]:
                edges_to_remove.add(edge_idx)
                # 只打印前几条，避免输出过多
                if len(edges_to_remove) <= 10 or edge_idx % 1000 == 0:
                    safe_print(f"找到需要合并的边组: 源节点名称='{src_name}', 目标节点名称='{dst_name}', 边索引={edge_idx}, 源节点ID={src_id}, 目标节点ID={dst_id}")
    
    show(f"找到 {len(nodes_to_merge)} 个需要合并的边组（源节点名称相同且目标节点名称相同）")
    
    if len(nodes_to_merge) == 0:
        show("未找到需要合并的节点，返回原始边列表")
        return edges
    
    # 第二遍遍历：重新连接边
    # 对于每个需要合并的节点对 (out_node, in_node)：
    # - 删除 out_node -> in_node 的边（已在 edges_to_remove 中）
    # - 将所有 out_node -> X 的边改为 in_node -> X
    # - 将所有 Y -> out_node 的边改为 Y -> in_node
    
    # 记录需要添加的新边（使用生成器模式或分批处理以减少内存占用）
    new_edges: List[Tuple[str, str, str, str, str, str]] = []
    # 限制新边列表大小，如果太大则分批写入
    MAX_NEW_EDGES_BATCH = 1000000  # 每批最多100万条新边
    # 记录需要修改的边索引：edge_index -> (new_src_id, new_dst_id)
    edges_to_modify: Dict[int, Tuple[str, str]] = {}
    # 记录已经处理过的边，避免重复处理
    processed_edges: Set[int] = set()
    # 使用 map 存储节点ID到代表节点ID的映射，快速查找
    node_to_representative: Dict[str, str] = {}  # node_id -> representative_node_id
    # 使用 map 存储已处理的节点，避免重复处理
    processed_nodes: Set[str] = set()  # 已处理的节点ID集合
    # 注意：不创建 edge_info_map 以节省内存，直接使用 edges 列表访问
    
    total_groups = len(nodes_to_merge)
    for group_idx, (representative_src_id, representative_dst_id, edge_list) in enumerate(nodes_to_merge, 1):
        # 获取代表节点的类型和名称
        rep_src_type = node_type_map.get(representative_src_id, "")
        rep_dst_type = node_type_map.get(representative_dst_id, "")
        rep_src_name = node_name_map.get(representative_src_id, "未知")
        rep_dst_name = node_name_map.get(representative_dst_id, "未知")
        
        # 打印正在处理的边组信息（带进度）
        if group_idx % 100 == 0 or group_idx == 1:
            safe_print(f"正在处理合并 [{group_idx}/{total_groups}]: 源节点名称='{rep_src_name}', 目标节点名称='{rep_dst_name}', 边组大小={len(edge_list)}")
        
        # 建立节点到代表节点的映射
        node_to_representative[representative_src_id] = representative_src_id
        node_to_representative[representative_dst_id] = representative_dst_id
        
        # 对于边组中除了代表边之外的其他边，需要合并它们的节点
        for src_id, dst_id, edge_idx in edge_list[1:]:  # 跳过第一条（代表边）
            # 建立映射：这些节点都映射到代表节点
            node_to_representative[src_id] = representative_src_id
            node_to_representative[dst_id] = representative_dst_id
            
            # 如果源节点不同且未处理过，需要将 src_id 的所有出边改为从 representative_src_id 发出
            if src_id != representative_src_id and src_id not in processed_nodes:
                processed_nodes.add(src_id)  # 标记为已处理
                # 处理 src_id 的所有出边：改为从 representative_src_id 发出
                out_edges_count = len(out_edges[src_id])
                if out_edges_count > 1000:
                    safe_print(f"  处理节点 {src_id} 的出边，共 {out_edges_count} 条...")
                for dst_id_out, edge_type, timestamp, edge_idx_out in out_edges[src_id]:
                    if edge_idx_out in edges_to_remove or edge_idx_out in processed_edges:
                        continue  # 跳过要删除的边和已处理的边
                    
                    # 直接从 edges 列表获取原始边的完整信息
                    orig_edge = edges[edge_idx_out]
                    src_id_orig, src_type, dst_id_orig, dst_type, edge_type_orig, timestamp_orig = orig_edge
                    
                    # 如果这条边是 src_id -> representative_dst_id，跳过（已在删除列表中）
                    if dst_id_orig == representative_dst_id:
                        continue
                    
                    # 创建新边：representative_src_id -> dst
                    new_src_type = rep_src_type if rep_src_type else src_type
                    new_edge = (representative_src_id, new_src_type, 
                               dst_id_orig, dst_type, edge_type_orig, timestamp_orig)
                    new_edges.append(new_edge)
                    processed_edges.add(edge_idx_out)
                    
                    # 如果新边列表太大，打印警告
                    if len(new_edges) > MAX_NEW_EDGES_BATCH:
                        safe_print(f"警告: 新边列表已超过 {MAX_NEW_EDGES_BATCH} 条，当前: {len(new_edges)}")
                
                # 处理指向 src_id 的所有入边：改为指向 representative_src_id
                in_edges_count = len(in_edges[src_id])
                if in_edges_count > 1000:
                    safe_print(f"  处理节点 {src_id} 的入边，共 {in_edges_count} 条...")
                for src_id_in, edge_type, timestamp, edge_idx_in in in_edges[src_id]:
                    if edge_idx_in in edges_to_remove or edge_idx_in in processed_edges:
                        continue  # 跳过要删除的边和已处理的边
                    
                    # 直接从 edges 列表获取原始边的完整信息
                    orig_edge = edges[edge_idx_in]
                    src_id_orig, src_type, dst_id_orig, dst_type, edge_type_orig, timestamp_orig = orig_edge
                    
                    # 如果这条边是 representative_src_id -> src_id，跳过（已在删除列表中）
                    if src_id_orig == representative_src_id:
                        continue
                    
                    # 修改边：src -> representative_src_id
                    edges_to_modify[edge_idx_in] = (src_id_orig, representative_src_id)
                    processed_edges.add(edge_idx_in)
            
            # 如果目标节点不同且未处理过，需要将 dst_id 的所有入边改为指向 representative_dst_id
            if dst_id != representative_dst_id and dst_id not in processed_nodes:
                processed_nodes.add(dst_id)  # 标记为已处理
                # 处理指向 dst_id 的所有入边：改为指向 representative_dst_id
                in_edges_count = len(in_edges[dst_id])
                if in_edges_count > 1000:
                    safe_print(f"  处理节点 {dst_id} 的入边，共 {in_edges_count} 条...")
                for src_id_in, edge_type, timestamp, edge_idx_in in in_edges[dst_id]:
                    if edge_idx_in in edges_to_remove or edge_idx_in in processed_edges:
                        continue  # 跳过要删除的边和已处理的边
                    
                    # 直接从 edges 列表获取原始边的完整信息
                    orig_edge = edges[edge_idx_in]
                    src_id_orig, src_type, dst_id_orig, dst_type, edge_type_orig, timestamp_orig = orig_edge
                    
                    # 如果这条边是 representative_src_id -> dst_id，跳过（已在删除列表中）
                    if src_id_orig == representative_src_id:
                        continue
                    
                    # 修改边：src -> representative_dst_id
                    edges_to_modify[edge_idx_in] = (src_id_orig, representative_dst_id)
                    processed_edges.add(edge_idx_in)
                
                # 处理 dst_id 的所有出边：改为从 representative_dst_id 发出
                out_edges_count = len(out_edges[dst_id])
                if out_edges_count > 1000:
                    safe_print(f"  处理节点 {dst_id} 的出边，共 {out_edges_count} 条...")
                for dst_id_out, edge_type, timestamp, edge_idx_out in out_edges[dst_id]:
                    if edge_idx_out in edges_to_remove or edge_idx_out in processed_edges:
                        continue  # 跳过要删除的边和已处理的边
                    
                    # 直接从 edges 列表获取原始边的完整信息
                    orig_edge = edges[edge_idx_out]
                    src_id_orig, src_type, dst_id_orig, dst_type, edge_type_orig, timestamp_orig = orig_edge
                    
                    # 如果这条边是 dst_id -> representative_src_id，跳过（已在删除列表中）
                    if dst_id_orig == representative_src_id:
                        continue
                    
                    # 创建新边：representative_dst_id -> dst
                    new_src_type = rep_dst_type if rep_dst_type else src_type
                    new_edge = (representative_dst_id, new_src_type, 
                               dst_id_orig, dst_type, edge_type_orig, timestamp_orig)
                    new_edges.append(new_edge)
                    processed_edges.add(edge_idx_out)
                    
                    # 如果新边列表太大，打印警告
                    if len(new_edges) > MAX_NEW_EDGES_BATCH:
                        safe_print(f"警告: 新边列表已超过 {MAX_NEW_EDGES_BATCH} 条，当前: {len(new_edges)}")
    
    # 构建结果边列表
    result_edges: List[Tuple[str, str, str, str, str, str]] = []
    
    safe_print(f"开始构建结果边列表，共 {len(edges)} 条原始边...")
    # 添加未删除且未修改的边
    for idx, edge in enumerate(edges):
        if idx % 100000 == 0 and idx > 0:
            safe_print(f"  已处理 {idx}/{len(edges)} 条边...")
        if idx in edges_to_remove:
            continue  # 跳过要删除的边
        
        if idx in edges_to_modify:
            # 修改边的目标节点
            new_src, new_dst = edges_to_modify[idx]
            src_id, src_type, dst_id, dst_type, edge_type, timestamp = edge
            # 修改目标节点为 in_node
            result_edges.append((src_id, src_type, new_dst, dst_type, edge_type, timestamp))
        else:
            result_edges.append(edge)
    
    # 添加新边（从 in_node 发出的边）
    result_edges.extend(new_edges)
    
    show(f"UnitMerge 完成：原始边数={len(edges)}, 删除边数={len(edges_to_remove)}, 新增边数={len(new_edges)}, 修改边数={len(edges_to_modify)}, 最终边数={len(result_edges)}")
    
    return result_edges

# 主处理流程
for path in path_list:
    show(f"开始处理: {path}")
    
    # 第一遍：提取节点信息（ID、类型、名称）
    id_nodetype_map: Dict[str, str] = {}
    id_nodename_map: Dict[str, str] = {}
    
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
                print(cnt)
            
            # 跳过 Event 和 Host 类型
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                continue
            
            if len(pattern_uuid.findall(line)) == 0:
                safe_print(f"警告: 未找到 UUID: {line[:100]}")
                continue
            
            uuid = pattern_uuid.findall(line)[0]
            subject_type = pattern_type.findall(line)
            
            # 提取节点类型
            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid] = 'MemoryObject'
                elif 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid] = 'NetFlowObject'
                elif 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid] = 'UnnamedPipeObject'
                else:
                    continue
            else:
                id_nodetype_map[uuid] = subject_type[0]
            
            # 提取节点名称
            node_name = extract_node_name(line)
            if node_name:
                id_nodename_map[uuid] = node_name
        
        f.close()
    
    show(f"节点信息提取完成: 类型映射={len(id_nodetype_map)}, 名称映射={len(id_nodename_map)}")
    
    # 第二遍：提取所有边
    all_edges: List[Tuple[str, str, str, str, str, str]] = []
    not_in_cnt = 0
    
    for i in range(100):
        now_path = path + '.' + str(i)
        if i == 0:
            now_path = path
        if not osp.exists(now_path):
            break
        
        f = open(now_path, 'r', encoding='utf-8')
        show(f"读取边信息: {now_path}")
        cnt = 0
        
        for line in f:
            cnt += 1
            if cnt % notice_num == 0:
                print(cnt)
            
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
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
                
                if srcId not in id_nodetype_map.keys():
                    not_in_cnt += 1
                    continue
                srcType = id_nodetype_map[srcId]
                
                # 处理第一个目标节点
                dstId1 = pattern_dst1.findall(line)
                if len(dstId1) > 0 and dstId1[0] != 'null':
                    dstId1 = dstId1[0]
                    if dstId1 not in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    dstType1 = id_nodetype_map[dstId1]
                    all_edges.append((srcId, srcType, dstId1, dstType1, edgeType, timestamp))
                
                # 处理第二个目标节点
                dstId2 = pattern_dst2.findall(line)
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if dstId2 not in id_nodetype_map.keys():
                        not_in_cnt += 1
                        continue
                    dstType2 = id_nodetype_map[dstId2]
                    all_edges.append((srcId, srcType, dstId2, dstType2, edgeType, timestamp))
        
        f.close()
    
    show(f"边信息提取完成: 总边数={len(all_edges)}, 缺失节点数={not_in_cnt}")
    
    # 执行 UnitMerge 算法
    show("开始执行 UnitMerge 算法...")
    merged_edges = unitmerge_process(all_edges, id_nodename_map, id_nodetype_map)
    
    # 按照 parse_darpatc.py 的逻辑，为每个分片文件创建对应的输出文件
    # 创建文件句柄映射（与 parse_darpatc.py 保持一致）
    file_handles: Dict[int, any] = {}
    for i in range(100):
        now_path = path + '.' + str(i)
        if i == 0:
            now_path = path
        if not osp.exists(now_path):
            break
        
        output_path = now_path + '.unitmerge.txt'
        file_handles[i] = open(output_path, 'w', encoding='utf-8')
    
    # 将合并后的边写入第一个文件（主文件）
    # 注意：由于 UnitMerge 是全局操作，合并后的边可能来自不同分片
    # 为了保持与 parse_darpatc.py 的文件结构一致，我们将所有合并后的边写入主文件
    if 0 in file_handles:
        for edge in merged_edges:
            edge_line = '\t'.join(edge) + '\n'
            file_handles[0].write(edge_line)
    
    # 关闭所有文件句柄
    for fw in file_handles.values():
        fw.close()
    
    show(f"处理完成: {path}，已创建 {len(file_handles)} 个输出文件")

# 复制文件到目标目录（根据需要修改）
os.system('cp ta1-theia-e3-official-1r.json.txt ../graphchi-cpp-master/graph_data/darpatc/theia_train.txt')
os.system('cp ta1-theia-e3-official-6r.json.8.txt ../graphchi-cpp-master/graph_data/darpatc/theia_test.txt')
os.system('cp ta1-cadets-e3-official.json.1.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt')
os.system('cp ta1-cadets-e3-official-2.json.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt')
os.system('cp ta1-fivedirections-e3-official-2.json.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_train.txt')
os.system('cp ta1-fivedirections-e3-official-2.json.23.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_test.txt')
os.system('cp ta1-trace-e3-official-1.json.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/trace_train_unitmerge.txt')
os.system('cp ta1-trace-e3-official-1.json.4.unitmerge.txt ../graphchi-cpp-master/graph_data/darpatc/trace_test_unitmerge.txt')
# os.system('rm ta1-*')


