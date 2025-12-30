import os.path as osp
import os
import argparse
import torch
import time
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler, DataLoader
from torch_geometric.nn import SAGEConv, GATConv
from torch.nn import CrossEntropyLoss
from sklearn.utils import class_weight
from data_process_train_balanced_weights import *
from data_process_test_balanced_weights import *

thre_map = {"cadets":1.5,"trace":1.0,"theia":1.5,"fivedirections":1.0}

def show(*s):
	for i in range(len(s)):
		print (str(s[i]) + ' ', end = '')
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


class SAGENet(torch.nn.Module):
	def __init__(self, in_channels, out_channels, concat=False):
		super(SAGENet, self).__init__()
		self.conv1 = SAGEConv(in_channels, 8, normalize=False, concat=concat)
		self.conv2 = SAGEConv(8, out_channels, normalize=False, concat=concat)

	def forward(self, x, data_flow):
		data = data_flow[0]
		x = x[data.n_id]
		x = F.relu(self.conv1((x, None), data.edge_index, size=data.size,res_n_id=data.res_n_id))
		x = F.dropout(x, p=0.5, training=self.training)
		data = data_flow[1]
		x = self.conv2((x, None), data.edge_index, size=data.size,res_n_id=data.res_n_id)

		# 返回原始logits，不使用log_softmax，因为CrossEntropyLoss内部会处理
		return x


def build_class_weights(train_labels, label_num, device):
	"""为现有类别计算balanced权重，并为缺失类别填充1.0，保证长度等于label_num。
	然后将权重映射到[0.5, 1.5]范围。"""
	unique_labels = np.unique(train_labels)
	weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=train_labels)
	full_weights = torch.ones(label_num, dtype=torch.float)
	for cls, w in zip(unique_labels, weights):
		full_weights[int(cls)] = float(w)
	
	# 将权重映射到[0.5, 1.5]范围
	# 只对非1.0的权重进行映射（即实际计算出的类别权重）
	computed_weights = full_weights[full_weights != 1.0]
	if len(computed_weights) > 0:
		min_weight = computed_weights.min().item()
		max_weight = computed_weights.max().item()
		
		if max_weight > min_weight:
			# 线性映射到[0.5, 1.5]
			# new_weight = 0.5 + (weight - min_weight) / (max_weight - min_weight) * (1.5 - 0.5)
			for i in range(label_num):
				if full_weights[i] != 1.0:  # 只映射计算出的权重
					full_weights[i] = 0.5 + (full_weights[i] - min_weight) / (max_weight - min_weight) * 1.0
		else:
			# 如果所有权重相同，设为1.0
			for i in range(label_num):
				if full_weights[i] != 1.0:
					full_weights[i] = 1.0
	
	return full_weights.to(device)

def train():
	model.train()
	total_loss = 0
	for data_flow in loader(data.train_mask):
		optimizer.zero_grad()
		out = model(data.x.to(device), data_flow.to(device))
		# 当前batch的标签
		target = data.y[data_flow.n_id].to(device)
		# 检查是否有节点权重（仅在启用边权重时才有）
		if hasattr(data, 'node_weight') and data.node_weight is not None:
			# 结合类别权重(class_weights)和节点权重(node_weight)做加权交叉熵
			sample_weight = data.node_weight[data_flow.n_id].to(device)
			per_sample_loss = F.cross_entropy(out, target, weight=class_weights, reduction='none')
			loss = (per_sample_loss * sample_weight).sum() / (sample_weight.sum() + 1e-8)
		else:
			# 没有节点权重时，只使用类别权重
			loss = F.cross_entropy(out, target, weight=class_weights, reduction='mean')
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * data_flow.batch_size
	return total_loss / data.train_mask.sum().item()

def test(mask):
	model.eval()
	correct = 0
	for data_flow in loader(mask):
		out = model(data.x.to(device), data_flow.to(device))
		# 模型输出原始logits，需要转换为log_softmax用于预测
		out_log = F.log_softmax(out, dim=1)
		pred = out_log.max(1)[1]
		# 从原始logits计算softmax概率
		pro  = F.softmax(out, dim=1)
		pro1 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			pro[i][pro1[1][i]] = -1
		pro2 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			if pro1[0][i]/pro2[0][i] < thre:
				pred[i] = 100
		correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
	return correct / mask.sum().item()

def final_test(mask):
	model.eval()
	correct = 0
	for data_flow in loader(mask):
		out = model(data.x.to(device), data_flow.to(device))
		# 模型输出原始logits，需要转换为log_softmax用于预测
		out_log = F.log_softmax(out, dim=1)
		pred = out_log.max(1)[1]
		# 从原始logits计算softmax概率
		pro  = F.softmax(out, dim=1)
		pro1 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			pro[i][pro1[1][i]] = -1
		pro2 = pro.max(1)
		for i in range(len(data_flow.n_id)):
			if pro1[0][i]/pro2[0][i] < thre:
				pred[i] = 100
		for i in range(len(data_flow.n_id)):
			if data.y[data_flow.n_id[i]] != pred[i]:
				fp.append(int(data_flow.n_id[i]))
			else:
				tn.append(int(data_flow.n_id[i]))
		correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
	return correct / mask.sum().item()

def validate():
	global fp, tn
	global loader, device, model, optimizer, data

	show('Start validating')
	path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_test.txt'
	data, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neibor = MyDatasetA(path, 0)
	dataset = TestDatasetA(data)
	data = dataset[0]
	print(data)
	loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False, add_self_loops=True)
	device = torch.device('cpu')	
	Net = SAGENet	
	model1 = Net(feature_num, label_num).to(device)
	model = model1
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	fp = []
	tn = []

	out_loop = -1
	while(1):
		out_loop += 1
		print('validating in model ', str(out_loop))
		model_path = '../models/model_'+str(out_loop)
		if not osp.exists(model_path): break
		model.load_state_dict(torch.load(model_path))
		fp = []
		tn = []
		auc = final_test(data.test_mask)
		print('fp and fn: ', len(fp), len(tn))
		_fp = 0
		_tp = 0
		eps = 1e-10
		tempNodeA = {}
		for i in nodeA:
			tempNodeA[i] = 1
		for i in fp:
			if not i in _nodeA:
				_fp += 1
			if not i in _neibor.keys():
				continue
			for j in _neibor[i]:
				if j in tempNodeA.keys():
					tempNodeA[j] = 0
		for i in tempNodeA.keys():
			if tempNodeA[i] == 0:
				_tp += 1
		print('Precision: ', _tp/(_tp+_fp))
		print('Recall: ', _tp/len(nodeA))
		if (_tp/len(nodeA) > 0.8) and (_tp/(_tp+_fp+eps) > 0.7):
			while (1):
				out_loop += 1
				model_path = '../models/model_'+str(out_loop)
				if not osp.exists(model_path): break
				os.system('rm ../models/model_'+str(out_loop))
				os.system('rm ../models/tn_feature_label_'+str(graphId)+'_'+str(out_loop)+'.txt')
				os.system('rm ../models/fp_feature_label_'+str(graphId)+'_'+str(out_loop)+'.txt')
			return 1
		if (_tp/len(nodeA) <= 0.8):
			return 0
		for j in tn:
			data.test_mask[j] = False
		
	return 0

def train_pro():
	global data, nodeA, _nodeA, _neibor, b_size, feature_num, label_num, graphId
	global model, loader, optimizer, device, fp, tn, loop_num, criterion, class_weights
	os.system('python setup.py')
	path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_train.txt'
	graphId = 0
	show('Start training graph ' + str(graphId))
	data1, feature_num, label_num, adj, adj2 = MyDataset(path, 0)
	dataset = TestDataset(data1)
	data = dataset[0]
	print(data)
	print('feature ', feature_num, '; label ', label_num)
	loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False, add_self_loops=True)
	device = torch.device('cpu')
	Net = SAGENet
	model = Net(feature_num, label_num).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

	# 计算balanced class weights
	train_labels = data.y[data.train_mask].cpu().numpy()
	class_weights = build_class_weights(train_labels, label_num, device)
	criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')
	print('Computed class weights:', class_weights)

	for epoch in range(1, 30):
		loss = train()
		auc = test(data.test_mask)
		show(epoch, loss, auc)

	loop_num = 0
	max_thre = 3
	bad_cnt = 0
	while (1):
		fp = []
		tn = []
		auc = final_test(data.test_mask)
		if len(tn) == 0:
			bad_cnt += 1
		else:
			bad_cnt = 0
		if bad_cnt >= max_thre:
			break

		if len(tn) > 0:
			for i in tn:
				data.train_mask[i] = False
				data.test_mask[i] = False

			fw = open('../models/fp_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt', 'w')
			x_list = data.x[fp]
			y_list = data.y[fp]
			print(len(x_list))
			
			if len(x_list) >1:
				sorted_index = np.argsort(y_list, axis = 0)
				x_list = np.array(x_list)[sorted_index]
				y_list = np.array(y_list)[sorted_index]

			for i in range(len(y_list)):
				fw.write(str(y_list[i])+':')
				for j in x_list[i]:
					fw.write(' '+str(j))
				fw.write('\n')
			fw.close()

			fw = open('../models/tn_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt', 'w')
			x_list = data.x[tn]
			y_list = data.y[tn]
			print(len(x_list))
			
			if len(x_list) >1:
				sorted_index = np.argsort(y_list, axis = 0)
				x_list = np.array(x_list)[sorted_index]
				y_list = np.array(y_list)[sorted_index]

			for i in range(len(y_list)):
				fw.write(str(y_list[i])+':')
				for j in x_list[i]:
					fw.write(' '+str(j))
				fw.write('\n')
			fw.close()
			torch.save(model.state_dict(),'../models/model_'+str(loop_num))
			loop_num += 1
			if len(fp) == 0: break
		auc = 0
		for epoch in range(1, 150):
			loss = train()
			auc = test(data.test_mask)
			show(epoch, loss, auc)
			if loss < 1: break
	show('Finish training graph ' + str(graphId))


def main():
	global b_size, args, thre
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='SAGE')
	parser.add_argument('--scene', type=str, default='theia')
	args = parser.parse_args()
	assert args.model in ['SAGE']
	assert args.scene in ['cadets','trace','theia','fivedirections']
	b_size = 5000
	thre = thre_map[args.scene]
	os.system('cp ../groundtruth/'+args.scene+'.txt groundtruth_uuid.txt')
	while (1):
		train_pro()
		flag = validate()
		if flag == 1:
			break
		else:
			os.system('rm ../models/model_*')
			os.system('rm ../models/tn_feature_label_*')
			os.system('rm ../models/fp_feature_label_*')



if __name__ == "__main__":
	graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
	os.environ['GRAPHCHI_ROOT'] = graphchi_root
	
	main()

