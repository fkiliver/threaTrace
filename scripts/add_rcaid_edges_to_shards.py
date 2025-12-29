import time
import os
import os.path as osp
import csv

def show(str):
    print(str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

# 文件映射：分片文件 -> CSV文件 -> 目标文件
file_mappings = [
    {
        'shard_file': 'ta1-theia-e3-official-1r.json.txt',
        'csv_file': 'supplementary-edge-rcaid/theia_train.csv',
        'output_file': 'ta1-theia-e3-official-1r.json.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/theia_train_rcaid.txt'
    },
    {
        'shard_file': 'ta1-theia-e3-official-6r.json.8.txt',
        'csv_file': 'supplementary-edge-rcaid/theia_test.csv',
        'output_file': 'ta1-theia-e3-official-6r.json.8.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/theia_test_rcaid.txt'
    },
    {
        'shard_file': 'ta1-cadets-e3-official.json.1.txt',
        'csv_file': 'supplementary-edge-rcaid/cadets_train.csv',
        'output_file': 'ta1-cadets-e3-official.json.1.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/cadets_train_rcaid.txt'
    },
    {
        'shard_file': 'ta1-cadets-e3-official-2.json.txt',
        'csv_file': 'supplementary-edge-rcaid/cadets_test.csv',
        'output_file': 'ta1-cadets-e3-official-2.json.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/cadets_test_rcaid.txt'
    },
    {
        'shard_file': 'ta1-trace-e3-official-1.json.txt',
        'csv_file': 'supplementary-edge-rcaid/trace_train.csv',
        'output_file': 'ta1-trace-e3-official-1.json.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/trace_train_rcaid.txt'
    },
    {
        'shard_file': 'ta1-trace-e3-official-1.json.4.txt',
        'csv_file': 'supplementary-edge-rcaid/trace_test.csv',
        'output_file': 'ta1-trace-e3-official-1.json.4.txt_rcaid',
        'copy_to': '../graphchi-cpp-master/graph_data/darpatc/trace_test_rcaid.txt'
    }
]

def process_shard_file(shard_file, csv_file, output_file):
    """
    处理分片文件，添加RCAID边
    
    :param shard_file: 分片文件路径
    :param csv_file: CSV文件路径（包含要添加的边）
    :param output_file: 输出文件路径
    :return: (节点数量, 添加的边数量)
    """
    # 规范化路径
    shard_file = osp.normpath(osp.abspath(shard_file))
    csv_file = osp.normpath(osp.abspath(csv_file))
    output_file = osp.normpath(osp.abspath(output_file))
    
    # 检查文件是否存在
    if not osp.exists(shard_file):
        raise FileNotFoundError(f"分片文件不存在: {shard_file}")
    if not osp.exists(csv_file):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
    
    show(f"开始处理: {shard_file}")
    
    # 第一步：读取分片文件，收集节点及其类型，同时保存所有原始边
    node_type_map = {}  # node_id -> node_type
    original_edges = []  # 保存所有原始边
    min_timestamp = None  # 记录最早的时间戳
    
    show("正在读取分片文件，收集节点信息...")
    with open(shard_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # 分片文件格式: srcId \t srcType \t dstId \t dstType \t edgeType \t timestamp
            parts = line_stripped.split('\t')
            if len(parts) != 6:
                continue
            
            src_id, src_type, dst_id, dst_type, edge_type, timestamp = parts
            
            # 保存节点类型
            node_type_map[src_id] = src_type
            node_type_map[dst_id] = dst_type
            
            # 保存原始边
            original_edges.append(line_stripped)
            
            # 记录最早的时间戳
            try:
                ts_int = int(timestamp)
                if min_timestamp is None or ts_int < min_timestamp:
                    min_timestamp = ts_int
            except ValueError:
                pass
    
    show(f"分片文件包含 {len(node_type_map)} 个节点，{len(original_edges)} 条原始边")
    
    # 第二步：读取CSV文件，找出两个节点都在分片文件中的边
    show(f"正在读取CSV文件: {csv_file}")
    rcaid_edges = []  # 存储要添加的边
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) < 2:
                continue
            
            src_id = row[0].strip()
            dst_id = row[1].strip()
            
            # 检查两个节点是否都在分片文件中
            if src_id in node_type_map and dst_id in node_type_map:
                src_type = node_type_map[src_id]
                dst_type = node_type_map[dst_id]
                edge_type = 'EVENT_RCAID'
                timestamp = str(min_timestamp) if min_timestamp is not None else '0'
                
                # 构建边字符串
                edge_line = f"{src_id}\t{src_type}\t{dst_id}\t{dst_type}\t{edge_type}\t{timestamp}"
                rcaid_edges.append(edge_line)
    
    show(f"找到 {len(rcaid_edges)} 条需要添加的RCAID边")
    
    # 第三步：将原始边和新边写入输出文件
    show(f"正在写入输出文件: {output_file}")
    
    # 确保输出目录存在
    output_dir = osp.dirname(output_file)
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 先写入所有原始边
        for edge in original_edges:
            f.write(edge + '\n')
        
        # 再写入所有RCAID边
        for edge in rcaid_edges:
            f.write(edge + '\n')
    
    show(f"处理完成: 原始边 {len(original_edges)} 条，新增RCAID边 {len(rcaid_edges)} 条")
    
    return len(node_type_map), len(rcaid_edges)

def main():
    show("开始处理所有分片文件...")
    
    total_nodes = 0
    total_edges_added = 0
    
    for mapping in file_mappings:
        try:
            shard_file = mapping['shard_file']
            csv_file = mapping['csv_file']
            output_file = mapping['output_file']
            copy_to = mapping['copy_to']
            
            # 处理文件
            nodes_count, edges_count = process_shard_file(shard_file, csv_file, output_file)
            total_nodes += nodes_count
            total_edges_added += edges_count
            
            # 复制文件到目标位置
            if copy_to:
                copy_to = osp.normpath(osp.abspath(copy_to))
                copy_to_dir = osp.dirname(copy_to)
                if copy_to_dir and not osp.exists(copy_to_dir):
                    os.makedirs(copy_to_dir, exist_ok=True)
                
                show(f"正在复制文件到: {copy_to}")
                os.system(f'cp {output_file} {copy_to}')
                show(f"复制完成: {copy_to}")
            
        except Exception as e:
            show(f"处理 {mapping['shard_file']} 时出错: {e}")
            continue
    
    show(f"所有处理完成！总共处理 {len(file_mappings)} 个文件，新增 {total_edges_added} 条RCAID边")

if __name__ == '__main__':
    main()

