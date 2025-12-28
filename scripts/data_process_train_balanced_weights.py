import os.path as osp
import argparse
import torch
import time
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data, InMemoryDataset

# 尝试从配置文件导入权重函数和开关，如果不存在则使用默认函数
try:
	from edge_weight_config import get_edge_weight, ENABLE_WEIGHT
except ImportError:
	# 如果配置文件不存在，使用默认权重函数和开关
	ENABLE_WEIGHT = False
	def get_edge_weight(edge_type_str):
		"""默认权重函数，所有边类型权重为1.0"""
		return 1.0

class TestDataset(InMemoryDataset):
	def __init__(self, data_list):
		super(TestDataset, self).__init__('/tmp/TestDataset')
		self.data, self.slices = self.collate(data_list)

	def _download(self):
		pass
	def _process(self):
		pass

def MyDataset(path, model):
	graphId = model
	node_cnt = 0
	nodeType_cnt = 0
	edgeType_cnt = 0
	provenance = []
	nodeType_map = {}
	edgeType_map = {}
	edge_s = []
	edge_e = []
	edge_weights = []  # 存储边权重
	edge_type_str_list = [] if ENABLE_WEIGHT else None  # 存储原始边类型字符串，用于计算权重（仅在启用权重时使用）
	data_thre = 1000000

	for out_loop in range(1):
		f = open(path, 'r')

		nodeId_map = {}

		for line in f:
			temp = line.strip('\n').split('\t')
			if not (temp[0] in nodeId_map.keys()):
				nodeId_map[temp[0]] = node_cnt
				node_cnt += 1
			temp[0] = nodeId_map[temp[0]]	

			if not (temp[2] in nodeId_map.keys()):
				nodeId_map[temp[2]] = node_cnt
				node_cnt += 1
			temp[2] = nodeId_map[temp[2]]

			if not (temp[1] in nodeType_map.keys()):
				nodeType_map[temp[1]] = nodeType_cnt
				nodeType_cnt += 1
			temp[1] = nodeType_map[temp[1]]

			if not (temp[3] in nodeType_map.keys()):
				nodeType_map[temp[3]] = nodeType_cnt
				nodeType_cnt += 1
			temp[3] = nodeType_map[temp[3]]
			
			edge_type_str = temp[4]  # 保存原始边类型字符串
			if not (edge_type_str in edgeType_map.keys()):
				edgeType_map[edge_type_str] = edgeType_cnt
				edgeType_cnt += 1

			temp[4] = edgeType_map[edge_type_str]
			edge_s.append(temp[0])
			edge_e.append(temp[2])
			if ENABLE_WEIGHT:
				edge_type_str_list.append(edge_type_str)  # 保存原始边类型字符串
			provenance.append(temp)

	f_train_feature = open('../models/feature.txt', 'w')
	for i in edgeType_map.keys():
		f_train_feature.write(str(i)+'\t'+str(edgeType_map[i])+'\n')
	f_train_feature.close()
	f_train_label = open('../models/label.txt', 'w')
	for i in nodeType_map.keys():
		f_train_label.write(str(i)+'\t'+str(nodeType_map[i])+'\n')
	f_train_label.close()
	feature_num = edgeType_cnt
	label_num = nodeType_cnt

	x_list = []
	y_list = []
	train_mask = []
	test_mask = []
	for i in range(node_cnt):
		temp_list = []
		for j in range(feature_num*2):
			temp_list.append(0)
		x_list.append(temp_list)
		y_list.append(0)
		train_mask.append(True)
		test_mask.append(True)
	for temp in provenance:
		srcId = temp[0]
		srcType = temp[1]
		dstId = temp[2]
		dstType = temp[3]
		edge = temp[4]
		x_list[srcId][edge] += 1
		y_list[srcId] = srcType
		x_list[dstId][edge+feature_num] += 1
		y_list[dstId] = dstType

	x = torch.tensor(x_list, dtype=torch.float)	
	y = torch.tensor(y_list, dtype=torch.long)
	train_mask = torch.tensor(train_mask, dtype=torch.bool)
	test_mask = train_mask
	edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
	
	# 节点权重初始化为1.0，当启用边权重时再根据边权重进行更新
	node_weight_list = [1.0 for _ in range(node_cnt)]
	
	# 根据开关决定是否计算和应用权重
	if ENABLE_WEIGHT:
		# 先根据边类型计算每条边的权重，同时统计每个节点相邻边的权重和与度数
		node_weight_sum = [0.0 for _ in range(node_cnt)]
		node_weight_deg = [0 for _ in range(node_cnt)]
		for (s, e), edge_type_str in zip(zip(edge_s, edge_e), edge_type_str_list):
			weight = get_edge_weight(edge_type_str)
			edge_weights.append(weight)
			# 统计两端节点的权重和与度数
			node_weight_sum[s] += weight
			node_weight_sum[e] += weight
			node_weight_deg[s] += 1
			node_weight_deg[e] += 1
		# 计算节点权重：使用相邻边权重的平均值（无边的节点保持1.0）
		for i in range(node_cnt):
			if node_weight_deg[i] > 0:
				node_weight_list[i] = node_weight_sum[i] / float(node_weight_deg[i])
		edge_weight = torch.tensor(edge_weights, dtype=torch.float)
		node_weight = torch.tensor(node_weight_list, dtype=torch.float)
		data1 = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, node_weight=node_weight, train_mask=train_mask, test_mask=test_mask)
	else:
		# 不使用边权重时，节点权重全部为1.0
		node_weight = torch.tensor(node_weight_list, dtype=torch.float)
		data1 = Data(x=x, y=y, edge_index=edge_index, node_weight=node_weight, train_mask=train_mask, test_mask=test_mask)
	feature_num *= 2
	return [data1], feature_num, label_num,0,0

