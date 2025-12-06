import time
import pandas as pd
import numpy as np
import os
import os.path as osp
import csv
import re
import json
import glob

def show(str):
	print (str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def parse_llm_edges(edge_dir, id_nodetype_map, default_timestamp='0'):
	"""
	从 scripts/edge/{dataset} 目录下的文件中解析 LLM 建议的边
	支持 trace、theia、cadets 等数据集
	
	:param edge_dir: 边文件目录路径（如 scripts/edge/trace, scripts/edge/theia, scripts/edge/cadets）
	:param id_nodetype_map: 节点ID到节点类型的映射
	:param default_timestamp: 默认时间戳（如果未指定）
	:return: 边列表，每个元素为 (srcId, srcType, dstId, dstType, edgeType, timestamp)
	"""
	llm_edges = []
	edge_files = glob.glob(osp.join(edge_dir, 'result_*.txt'))
	
	if not edge_files:
		show(f"警告: 在 {edge_dir} 中未找到边文件")
		return llm_edges
	
	show(f"开始解析 LLM 边文件，共找到 {len(edge_files)} 个文件")
	
	for edge_file in edge_files:
		try:
			with open(edge_file, 'r', encoding='utf-8') as f:
				content = f.read()
			
			# 查找 "--- 回答 ---" 分隔符
			if '--- 回答 ---' not in content:
				continue
			
			# 提取 JSON 部分
			# 首先查找 ```json 代码块
			json_start = content.find('```json')
			if json_start != -1:
				# 找到代码块开始，查找第一个 [ 或 {
				json_start = content.find('[', json_start)
				if json_start == -1:
					json_start = content.find('{', content.find('```json'))
				json_end = content.find('```', json_start + 1)
				if json_end != -1:
					# 在代码块结束前查找最后一个 ] 或 }
					json_end = content.rfind(']', json_start, json_end)
					if json_end == -1:
						json_end = content.rfind('}', json_start, json_end)
					if json_end != -1:
						json_end += 1
			else:
				# 没有代码块，直接查找 JSON 开始位置
				json_start = content.find('[')
				if json_start == -1:
					json_start = content.find('{')
				
				if json_start != -1:
					# 从开始位置查找匹配的结束位置
					# 简单方法：从后往前查找最后一个 ] 或 }
					json_end = content.rfind(']')
					if json_end == -1 or json_end < json_start:
						json_end = content.rfind('}')
					if json_end != -1 and json_end >= json_start:
						json_end += 1
			
			if json_start == -1 or json_end == -1 or json_end <= json_start:
				continue
			
			json_str = content[json_start:json_end]
			
			# 解析 JSON
			try:
				parsed_json = json.loads(json_str)
			except json.JSONDecodeError as e:
				show(f"警告: 解析 {edge_file} 的 JSON 失败: {e}")
				continue
			
			# 处理两种格式：
			# 1. 直接数组格式：[{"subject": "...", "object": "...", "confidence level": ...}]
			# 2. 对象格式：{"causal_relations": [{"subject": "...", "object": "...", "confidence_level": ...}]}
			edges_data = []
			if isinstance(parsed_json, list):
				# 直接数组格式
				edges_data = parsed_json
			elif isinstance(parsed_json, dict) and 'causal_relations' in parsed_json:
				# 对象格式，提取 causal_relations 字段
				edges_data = parsed_json['causal_relations']
			
			# 处理每条边
			for edge_data in edges_data:
				# 处理两种字段名：'confidence level' 和 'confidence_level'
				subject = edge_data.get('subject', '').strip()
				object_node = edge_data.get('object', '').strip()
				confidence = edge_data.get('confidence level', edge_data.get('confidence_level', 0))
				
				if not subject or not object_node:
					continue
				
				# 检查节点是否在 id_nodetype_map 中
				if subject not in id_nodetype_map or object_node not in id_nodetype_map:
					continue
				
				srcType = id_nodetype_map[subject]
				dstType = id_nodetype_map[object_node]
				edgeType = 'EVENT_LLM_SUGGESTED'
				# 使用传入的默认时间戳
				timestamp = default_timestamp
				
				llm_edges.append((subject, srcType, object_node, dstType, edgeType, timestamp))
		
		except Exception as e:
			show(f"警告: 处理文件 {edge_file} 时出错: {e}")
			continue
	
	show(f"成功解析 {len(llm_edges)} 条 LLM 建议的边")
	return llm_edges

os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official.json.tar.gz')
os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-cadets-e3-official-2.json.tar.gz')
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-fivedirections-e3-official-2.json.tar.gz')
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-theia-e3-official-1r.json.tar.gz')
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-theia-e3-official-6r.json.tar.gz')
# os.system('tar -zxvf ../graphchi-cpp-master/graph_data/darpatc/ta1-trace-e3-official-1.json.tar.gz')

path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json']
# path_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json' , 'ta1-trace-e3-official-1.json']
# path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json', 'ta1-fivedirections-e3-official-2.json', 'ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json', 'ta1-trace-e3-official-1.json']

pattern_uuid = re.compile(r'uuid\":\"(.*?)\"') 
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')

notice_num = 1000000

for path in path_list:
	id_nodetype_map = {}
	for i in range(100):
		now_path  = path + '.' + str(i)
		if i == 0: now_path = path
		if not osp.exists(now_path): break
		f = open(now_path, 'r', encoding='utf-8')
		show(now_path)
		cnt  = 0
		for line in f:
			cnt += 1
			if cnt % notice_num == 0:
				print(cnt)
			if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
			if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
			if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
			if len(pattern_uuid.findall(line)) == 0: print (line)
			uuid = pattern_uuid.findall(line)[0]
			subject_type = pattern_type.findall(line)

			if len(subject_type) < 1:
				if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
					id_nodetype_map[uuid] = 'MemoryObject'
					continue
				if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
					id_nodetype_map[uuid] = 'NetFlowObject'
					continue
				if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
					id_nodetype_map[uuid] = 'UnnamedPipeObject'
					continue

			id_nodetype_map[uuid] = subject_type[0]
		f.close()
	not_in_cnt = 0
	min_timestamp = None  # 记录最早的时间戳，用于 LLM 边
	for i in range(100):
		now_path  = path + '.' + str(i)
		if i == 0: now_path = path
		if not osp.exists(now_path): break
		f = open(now_path, 'r', encoding='utf-8')
		fw = open(now_path+'.txt', 'w', encoding='utf-8')
		cnt = 0
		for line in f:
			cnt += 1
			if cnt % notice_num == 0:
				print(cnt) 

			if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
				pattern = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
				edgeType = pattern_type.findall(line)[0]
				timestamp = pattern_time.findall(line)[0]
				# 记录最早的时间戳
				try:
					ts_int = int(timestamp)
					if min_timestamp is None or ts_int < min_timestamp:
						min_timestamp = ts_int
				except:
					pass
				srcId = pattern_src.findall(line)
				if len(srcId) == 0: continue
				srcId = srcId[0]
				if not srcId in id_nodetype_map.keys(): 
					not_in_cnt += 1
					continue
				srcType = id_nodetype_map[srcId]
				dstId1 = pattern_dst1.findall(line)
				if len(dstId1) > 0  and dstId1[0] != 'null':
					dstId1 = dstId1[0]
					if not dstId1 in id_nodetype_map.keys():
						not_in_cnt += 1
						continue
					dstType1 = id_nodetype_map[dstId1]
					this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
					fw.write(this_edge1)

				dstId2 = pattern_dst2.findall(line)
				if len(dstId2) > 0  and dstId2[0] != 'null':
					dstId2 = dstId2[0]
					if not dstId2 in id_nodetype_map.keys():
						not_in_cnt += 1
						continue
					dstType2 = id_nodetype_map[dstId2]
					this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
					fw.write(this_edge2)	
		fw.close()
		f.close()
	
	# 根据数据集类型添加 LLM 建议的边（在所有分片文件处理完成后）
	dataset_name = None
	path_lower = path.lower()
	if 'trace' in path_lower:
		dataset_name = 'trace'
	elif 'theia' in path_lower:
		dataset_name = 'theia'
	elif 'cadets' in path_lower:
		dataset_name = 'cadets'
	
	if dataset_name:
		edge_dir = osp.join('scripts', 'edge', dataset_name)
		if osp.exists(edge_dir):
			show(f"为 {path} 添加 LLM 建议的边（数据集: {dataset_name}）")
			# 使用最早的时间戳，如果没有则使用 0
			default_timestamp = str(min_timestamp) if min_timestamp is not None else '0'
			llm_edges = parse_llm_edges(edge_dir, id_nodetype_map, default_timestamp)
			if llm_edges:
				# 将 LLM 边追加到所有分片文件中
				for i in range(100):
					now_path = path + '.' + str(i) + '.txt'
					if i == 0:
						now_path = path + '.txt'
					if not osp.exists(now_path):
						break
					with open(now_path, 'a', encoding='utf-8') as fw:
						for edge in llm_edges:
							edge_line = '\t'.join(edge) + '\n'
							fw.write(edge_line)
				show(f"已将 {len(llm_edges)} 条 LLM 边添加到 {path} 的所有分片文件")
		else:
			show(f"警告: 边目录 {edge_dir} 不存在，跳过添加 LLM 边")
	
os.system('cp ta1-theia-e3-official-1r.json.txt ../graphchi-cpp-master/graph_data/darpatc/theia_train.txt')
os.system('cp ta1-theia-e3-official-6r.json.8.txt ../graphchi-cpp-master/graph_data/darpatc/theia_test.txt')
os.system('cp ta1-cadets-e3-official.json.1.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_train.txt')
os.system('cp ta1-cadets-e3-official-2.json.txt ../graphchi-cpp-master/graph_data/darpatc/cadets_test.txt')
os.system('cp ta1-fivedirections-e3-official-2.json.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_train.txt')
os.system('cp ta1-fivedirections-e3-official-2.json.23.txt ../graphchi-cpp-master/graph_data/darpatc/fivedirections_test.txt')
os.system('cp ta1-trace-e3-official-1.json.txt ../graphchi-cpp-master/graph_data/darpatc/trace_train.txt')
os.system('cp ta1-trace-e3-official-1.json.4.txt ../graphchi-cpp-master/graph_data/darpatc/trace_test.txt')

# os.system('rm ta1-*')

