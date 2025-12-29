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
from data_process_train import *
from data_process_test import *

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

		return F.log_softmax(x, dim=1)

def train():
	model.train()
	total_loss = 0
	for data_flow in loader(data.train_mask):
		optimizer.zero_grad()
		out = model(data.x.to(device), data_flow.to(device))
		loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * data_flow.batch_size
	return total_loss / data.train_mask.sum().item()

def test(mask):
	model.eval()
	correct = 0
	for data_flow in loader(mask):
		out = model(data.x.to(device), data_flow.to(device))
		pred = out.max(1)[1]
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
		pred = out.max(1)[1]
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
	path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_test_rcaid.txt'
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
	global model, loader, optimizer, device, fp, tn, loop_num
	os.system('python setup.py')
	path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_train_rcaid.txt'
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

	for epoch in range(1, 30):
		loss = train()
		auc = test(data.test_mask)
		show(epoch, loss, auc)

	loop_num = 0
	max_thre = 3
	bad_cnt = 0
	low_auc_cnt = 0
	min_auc_threshold = 0.01
	best_auc = 0.0
	best_loop_num = -1
	last_saved_auc = 0.0  # 上一次保存模型时的auc
	auc_drop_threshold = 0.3  # auc下降超过30%认为是暴跌
	should_stop_all_training = False  # 标志：是否应该结束整个训练大循环
	while (1):
		fp = []
		tn = []
		auc = final_test(data.test_mask)
		
		# 初始化best_auc（第一次循环）
		if best_auc == 0 and auc > 0:
			best_auc = auc
		
		# 检查auc是否暴跌（与之前最好的模型相比）
		if best_auc > 0:
			auc_drop_ratio = (best_auc - auc) / best_auc if best_auc > 0 else 0
			if auc_drop_ratio > auc_drop_threshold and auc < best_auc * 0.7:
				show('AUC dropped significantly! Current:', auc, 'Best:', best_auc, 'Drop ratio:', auc_drop_ratio)
				show('Rolling back to best model at loop', best_loop_num)
				# 回退到最好的模型
				if best_loop_num >= 0 and osp.exists('../models/model_'+str(best_loop_num)):
					model.load_state_dict(torch.load('../models/model_'+str(best_loop_num)))
					show('Model rolled back successfully')
					# 删除当前差的模型（如果已保存）
					if loop_num > best_loop_num:
						if osp.exists('../models/model_'+str(loop_num)):
							os.system('rm ../models/model_'+str(loop_num))
						if osp.exists('../models/tn_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt'):
							os.system('rm ../models/tn_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt')
						if osp.exists('../models/fp_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt'):
							os.system('rm ../models/fp_feature_label_'+str(graphId)+'_'+str(loop_num)+'.txt')
					# 重新测试以获取正确的fp和tn
					fp = []
					tn = []
					auc = final_test(data.test_mask)
					show('After rollback, AUC:', auc)
					# 回退后，使用回退的模型作为基准
					best_auc = auc
					last_saved_auc = auc
					# 继续使用当前的loop_num，但模型已经回退到best_loop_num的状态
		
		if len(tn) == 0:
			bad_cnt += 1
		else:
			bad_cnt = 0
		if bad_cnt >= max_thre:
			break
		
		# 检查auc是否持续很低
		if auc < min_auc_threshold:
			low_auc_cnt += 1
		else:
			low_auc_cnt = 0
		if low_auc_cnt >= max_thre:
			show('AUC too low, stopping training. Final AUC:', auc)
			show('Will stop entire training loop due to persistently low AUC')
			should_stop_all_training = True
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
			# 记录保存模型时的auc
			last_saved_auc = auc
			if auc > best_auc:
				best_auc = auc
				best_loop_num = loop_num
			loop_num += 1
			if len(fp) == 0: break
		
		# 保存训练开始时的模型状态，用于回退
		training_start_state = model.state_dict().copy()
		# 测试训练开始前的auc作为基准
		training_start_auc = test(data.test_mask)
		show('Training start AUC:', training_start_auc)
		
		auc = 0
		training_best_auc = training_start_auc
		training_best_epoch = 0
		training_best_state = training_start_state.copy()
		low_auc_epochs = 0  # 连续低auc的epoch数
		total_low_auc_epochs = 0  # 总共有多少个epoch的auc很低
		min_auc_for_training = 0.01  # 训练过程中最低可接受的auc
		should_stop_all_training = False  # 标志：是否应该结束整个训练大循环
		
		for epoch in range(1, 150):
			loss = train()
			auc = test(data.test_mask)
			
			# 跟踪训练过程中的最佳auc
			if auc > training_best_auc:
				training_best_auc = auc
				training_best_epoch = epoch
				training_best_state = model.state_dict().copy()
			
			# 检查auc是否很低，并累计计数
			if auc < min_auc_for_training:
				low_auc_epochs += 1
				total_low_auc_epochs += 1
			else:
				# auc在可接受范围内，重置连续计数
				low_auc_epochs = 0
			
			# 检查训练过程中是否暴跌（相对于训练开始时的auc）
			if training_start_auc > 0:
				auc_drop_ratio = (training_start_auc - auc) / training_start_auc if training_start_auc > 0 else 0
				if auc_drop_ratio > auc_drop_threshold and auc < training_start_auc * 0.7:
					show('Training AUC dropped significantly during training! Epoch:', epoch, 'Current:', auc, 'Start:', training_start_auc, 'Drop ratio:', auc_drop_ratio)
					show('Rolling back to best model during training at epoch', training_best_epoch, 'with AUC:', training_best_auc)
					# 回退到训练过程中的最佳模型
					if training_best_state is not None:
						model.load_state_dict(training_best_state)
						auc = training_best_auc
						show('Model rolled back to epoch', training_best_epoch, 'AUC:', auc)
						break  # 提前结束训练
			
			# 检查auc是否持续很低（即使训练开始时auc为0）
			if auc < min_auc_for_training:
				# 如果训练开始时的auc也很低，且连续多个epoch都很低，停止训练
				if training_start_auc < min_auc_for_training and low_auc_epochs >= 5:
					show('AUC persistently low during training! Epoch:', epoch, 'AUC:', auc, 'Start AUC:', training_start_auc, 'Consecutive low AUC epochs:', low_auc_epochs)
					show('Stopping training due to persistently low AUC')
					# 如果训练过程中有更好的auc，使用它
					if training_best_auc > auc and training_best_state is not None:
						show('Using best model from training: epoch', training_best_epoch, 'AUC:', training_best_auc)
						model.load_state_dict(training_best_state)
						auc = training_best_auc
					# 如果最佳auc仍然很低，标记为需要结束整个训练
					if training_best_auc < min_auc_for_training:
						should_stop_all_training = True
						show('Best AUC during training is still very low, will stop entire training loop')
					break
				# 如果训练开始时auc较高，但训练过程中暴跌到很低，也停止
				elif training_start_auc >= min_auc_for_training and low_auc_epochs >= 3:
					show('AUC dropped to very low during training! Epoch:', epoch, 'Current AUC:', auc, 'Start AUC:', training_start_auc, 'Consecutive low AUC epochs:', low_auc_epochs)
					show('Rolling back to best model during training at epoch', training_best_epoch, 'with AUC:', training_best_auc)
					if training_best_state is not None:
						model.load_state_dict(training_best_state)
						auc = training_best_auc
						show('Model rolled back to epoch', training_best_epoch, 'AUC:', auc)
					break
			
			# 如果训练开始时auc很低，且大部分epoch的auc都很低，也停止训练
			if training_start_auc < min_auc_for_training and epoch >= 10:
				low_auc_ratio = total_low_auc_epochs / epoch
				if low_auc_ratio >= 0.8:  # 80%以上的epoch auc都很低
					show('Most epochs have low AUC during training! Epoch:', epoch, 'Low AUC ratio:', low_auc_ratio, 'Total low AUC epochs:', total_low_auc_epochs)
					show('Stopping training due to majority of epochs having low AUC')
					# 如果训练过程中有更好的auc，使用它
					if training_best_auc > auc and training_best_state is not None:
						show('Using best model from training: epoch', training_best_epoch, 'AUC:', training_best_auc)
						model.load_state_dict(training_best_state)
						auc = training_best_auc
					# 如果最佳auc仍然很低，标记为需要结束整个训练
					if training_best_auc < min_auc_for_training:
						should_stop_all_training = True
						show('Best AUC during training is still very low, will stop entire training loop')
					break
			
			show(epoch, loss, auc)
			if loss < 1: 
				break
		
		# 训练结束后，使用训练过程中的最佳模型状态
		if training_best_state is not None and training_best_auc > auc:
			show('Using best model from training: epoch', training_best_epoch, 'AUC:', training_best_auc, 'instead of final AUC:', auc)
			model.load_state_dict(training_best_state)
			auc = training_best_auc
		
		# 检查训练结束后的最终auc是否持续很低
		if auc < min_auc_for_training and training_start_auc < min_auc_for_training:
			show('Final AUC after training is still very low:', auc, 'Training start AUC:', training_start_auc)
			show('Will stop entire training loop due to persistently low AUC')
			should_stop_all_training = True
		
		if loss < 1:
			break
		
		# 如果标记为需要结束整个训练，跳出外层循环
		if should_stop_all_training:
			show('Stopping entire training loop due to persistently low AUC')
			break
	show('Finish training graph ' + str(graphId))
	return should_stop_all_training


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
		stop_all = train_pro()
		if stop_all:
			show('Stopping entire training due to persistently low AUC')
			break
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
