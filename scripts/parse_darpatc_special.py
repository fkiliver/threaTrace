#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_darpatc_special.py

功能：在已经生成好的 DARPA TC 边列表文件上，找出"第一条边是出边"的节点及其所有后代节点，
并将根节点与后代节点直接相连的新边添加到原始边文件中。

输入边文件格式（与原始 *.txt 边文件一致）：
    src_id \t src_type \t dst_id \t dst_type \t edge_type \t timestampNanos
"""

import os
import argparse
from typing import Dict, List, Set, Tuple


def safe_print(*args, **kwargs) -> None:
    """
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


def find_descendants_dfs(start_node: str, graph_edges: Dict[str, List[str]]) -> Set[str]:
    """
    使用深度优先搜索查找节点的所有后代。

    :param start_node: 起始节点 ID
    :param graph_edges: 邻接表形式的图 {src: [dst1, dst2, ...]}
    :return: 所有后代节点 ID 的集合（不包含起始节点本身）
    """
    descendants: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = [start_node]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        # 跳过起始节点本身
        if current != start_node:
            descendants.add(current)

        # 添加所有直接后代到栈中
        for neighbor in graph_edges.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)

    return descendants


def find_special_nodes_and_add_edges(
    input_file: str, output_file: str, new_edge_type: str = "EVENT_FLASH", new_timestamp: int = None
) -> Tuple[int, int, int]:
    """
    读取边列表文件，找出特殊节点及其后代，并将根节点与后代节点直接相连的新边添加到原始边文件中。

    特殊节点定义：节点的第一条边是"出边"的节点。

    :param input_file: 输入边列表文件路径
    :param output_file: 输出边列表文件路径（包含原始边和新添加的边）
    :param new_edge_type: 新添加边的类型（默认：EVENT_FLASH）
    :param new_timestamp: 新添加边的时间戳（如果为None，则使用特殊节点的第一条边的时间戳）
    :return: (总行数, 特殊节点个数, 新添加的边数)
    """

    # 存储每个节点的第一条边信息
    # node_first_edges[node_id] = {'timestamp': int, 'is_out_edge': bool, 'edge_type': str, 'node_type': str}
    node_first_edges: Dict[str, Dict[str, object]] = {}
    # 存储图的边关系（用于查找后代）
    graph_edges: Dict[str, List[str]] = {}
    # 存储节点类型映射 node_id -> node_type
    node_type_map: Dict[str, str] = {}
    # 存储所有原始边
    original_edges: List[str] = []

    safe_print(f"正在读取文件: {input_file}")

    total_lines = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            line_stripped = line.strip()
            if not line_stripped:
                continue

            parts = line_stripped.split("\t")
            if len(parts) != 6:
                safe_print(f"警告: 第{line_num}行格式不正确，跳过")
                continue

            src_id, src_type, dst_id, dst_type, edge_type, timestamp = parts

            # 保存原始边
            original_edges.append(line_stripped)

            # 保存节点类型
            node_type_map[src_id] = src_type
            node_type_map[dst_id] = dst_type

            try:
                # 将时间戳转换为整数以便比较
                timestamp_int = int(timestamp)
            except ValueError:
                safe_print(f"警告: 第{line_num}行时间戳无法转换为整数，跳过")
                continue

            # 处理源节点（出边）
            if src_id not in node_first_edges:
                node_first_edges[src_id] = {
                    "timestamp": timestamp_int,
                    "is_out_edge": True,  # 作为源节点出现，是出边
                    "edge_type": edge_type,
                    "node_type": src_type,
                }
            else:
                # 如果找到更早的时间戳，更新记录
                if timestamp_int < int(node_first_edges[src_id]["timestamp"]):
                    node_first_edges[src_id] = {
                        "timestamp": timestamp_int,
                        "is_out_edge": True,
                        "edge_type": edge_type,
                        "node_type": src_type,
                    }

            # 处理目标节点（入边）
            if dst_id not in node_first_edges:
                node_first_edges[dst_id] = {
                    "timestamp": timestamp_int,
                    "is_out_edge": False,  # 作为目标节点出现，是入边
                    "edge_type": edge_type,
                    "node_type": dst_type,
                }
            else:
                # 如果找到更早的时间戳，更新记录
                if timestamp_int < int(node_first_edges[dst_id]["timestamp"]):
                    node_first_edges[dst_id] = {
                        "timestamp": timestamp_int,
                        "is_out_edge": False,
                        "edge_type": edge_type,
                        "node_type": dst_type,
                    }

            # 构建图的边关系（用于查找后代）
            if src_id not in graph_edges:
                graph_edges[src_id] = []
            if dst_id not in graph_edges:
                graph_edges[dst_id] = []

            # 添加出边关系
            graph_edges[src_id].append(dst_id)

    safe_print(f"成功处理 {total_lines} 行原始边数据")

    # 找出特殊节点（第一条边是出边的节点）
    special_nodes = [nid for nid, info in node_first_edges.items() if info["is_out_edge"]]
    safe_print(f"找到 {len(special_nodes)} 个特殊节点（根节点）")

    # 查找每个特殊节点的后代
    safe_print("正在查找特殊节点的后代...")
    new_edges: List[str] = []
    edges_added = 0

    for idx, special_node in enumerate(special_nodes, 1):
        if idx % 1000 == 0:
            safe_print(f"  已处理特殊节点 {idx}/{len(special_nodes)}")
        descendants = find_descendants_dfs(special_node, graph_edges)
        
        # 获取特殊节点的类型和时间戳
        special_node_type = node_first_edges[special_node]["node_type"]
        special_node_timestamp = node_first_edges[special_node]["timestamp"]
        
        # 为新边确定时间戳
        edge_timestamp = new_timestamp if new_timestamp is not None else special_node_timestamp
        
        for descendant in descendants:
            # 获取后代节点的类型
            if descendant not in node_type_map:
                safe_print(f"警告: 后代节点 {descendant} 的类型信息缺失，跳过")
                continue
            descendant_type = node_type_map[descendant]
            
            # 创建新边：根节点 -> 后代节点
            new_edge = f"{special_node}\t{special_node_type}\t{descendant}\t{descendant_type}\t{new_edge_type}\t{edge_timestamp}\n"
            new_edges.append(new_edge)
            edges_added += 1

    safe_print(f"找到 {edges_added} 个特殊节点-后代关系，将添加 {edges_added} 条新边")

    # 确保输出目录存在
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 写入输出文件：先写入原始边，再写入新边
    safe_print(f"正在写入结果到: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入所有原始边
        for edge in original_edges:
            f.write(edge + "\n")
        # 写入新添加的边
        for edge in new_edges:
            f.write(edge)

    safe_print(f"处理完成！共写入 {len(original_edges)} 条原始边和 {edges_added} 条新边")

    return total_lines, len(special_nodes), edges_added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 DARPA TC 边列表文件中找出第一条边为出边的节点及其所有后代，并将根节点与后代节点直接相连。"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="../graphchi-cpp-master/graph_data/darpatc/fivedirections_train.txt",
        help="输入边列表文件路径（默认：fivedirections_train.txt）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./flash-add-edge/fivedirections_train_with_flash.txt",
        help="输出边列表文件路径（包含原始边和新添加的边，默认：./flash-add-edge/fivedirections_train_with_flash.txt）",
    )
    parser.add_argument(
        "-e",
        "--edge-type",
        default="EVENT_FLASH",
        help="新添加边的类型（默认：EVENT_FLASH）",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=int,
        default=None,
        help="新添加边的时间戳（如果未指定，则使用根节点的第一条边的时间戳）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = args.input
    output_file = args.output
    new_edge_type = args.edge_type
    new_timestamp = args.timestamp

    try:
        total, special_cnt, new_edges_cnt = find_special_nodes_and_add_edges(
            input_file, output_file, new_edge_type, new_timestamp
        )
        safe_print(
            f"汇总：原始边数={total}，特殊节点（根节点）数={special_cnt}，新添加边数={new_edges_cnt}"
        )
    except FileNotFoundError:
        safe_print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        safe_print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


