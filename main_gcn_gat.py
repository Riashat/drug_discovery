import csv
import numpy as np
from matplotlib import pyplot as plt
#@title Importing RDKit for molecule parsing
import rdkit as rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from deepchem.feat import MolGraphConvFeaturizer as MGCF
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
import math
import torch
import torch.nn as nn
import random
import argparse
import os
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from models import GCN_GAT, GCN, GAT, GraphAttentionLayer, GraphMLP
from utils import train_val_test, load_data, binary_acc
from graph_featurizer import construct_adjacency_matrix, construct_graphs

def evaluate_model(args, model, all_test_smiles, test_logDs):
	model.eval()
	criterion = torch.nn.BCELoss() 

	all_predictions = []
	all_targets = []

	for idx, smiles in enumerate(all_test_smiles) : 
		
		test_const_adj_mat = smiles[1]
		test_node_features = smiles[0].node_features
		test_edge_index = smiles[0].edge_index
		test_labels = smiles[2]
		
		test_node_features = torch.Tensor(test_node_features)
		test_const_adj_mat = torch.Tensor(test_const_adj_mat)
		test_edge_index = torch.Tensor(test_edge_index).long()

		if args.model == 'GCN' or args.model == 'GCN_GAT':
			_, prediction_softmax, predicted_label = model(test_node_features, test_edge_index )
		elif args.model == 'GAT':
			_, prediction_softmax, predicted_label = model(test_node_features, test_const_adj_mat )
		else:
			_, prediction_softmax, predicted_label = model(test_node_features, test_edge_index )

		all_predictions.append(predicted_label[0])
		all_targets.append(int(test_labels))

	# Test accuracy
	y_true = np.array(all_targets)
	y_pred = np.array(all_predictions)
	acc = accuracy_score(y_true, y_pred)

	epoch_roc_auc = roc_auc_score(y_true, y_pred)

	return acc, epoch_roc_auc


def train_and_evaluation(args, epochs, model, all_smiles, labels, batch_size, eval_interval, all_test_smiles, test_logDs):
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = model.optimizer
	model.train()

	epoch_train_loss = []
	epoch_acc = []
	epoch_auc = []

	for epoch in range(epochs+1):
		print ("Train Epoch", epoch)
		optimizer.zero_grad()
		mini_batch_smiles = random.sample(all_smiles, batch_size)

		all_train_loss = 0.0

		for batch, adj_mat, labels in mini_batch_smiles : 
			node_features = batch.node_features
			edge_index = batch.edge_index

			node_features = torch.Tensor(node_features)
			edge_index = torch.Tensor(edge_index).long()
			adj_mat = torch.Tensor(adj_mat)

			if args.model == 'GCN' or args.model == 'GCN_GAT':
				_, out, predicted_label = model(node_features, edge_index )
			elif args.model == 'GAT' : ## GAT
				_, out, predicted_label = model(node_features, adj_mat )
			else :
				_, out, predicted_label = model(node_features, edge_index)
			labels = (torch.Tensor(np.array([labels])).long()).repeat(out.shape[0])

			loss = criterion(out,labels)

			all_train_loss += loss

		all_train_loss.backward()
		optimizer.step()

		epoch_train_loss.append(all_train_loss.data.numpy() )

		print ("Train Epoch Losses : ", epoch_train_loss[-1]) 

		if epoch % eval_interval==0 : 
			acc, epoch_roc_auc = evaluate_model(args, model, all_test_smiles, test_logDs)
			print("Eval Acc", acc)
			print("ROC AUC", epoch_roc_auc)
			epoch_acc.append(acc)
			epoch_auc.append(epoch_roc_auc)

	return epoch_train_loss, epoch_acc, epoch_auc


def main(args):
	## some default arguments : 
	hidden = 128
	batch_size = args.batch_size
	iterations = args.iterations
	eval_interval = args.eval_interval

	train_smiles, train_logDs, test_smiles, test_logDs = load_data(active=args.active_index)
	# train_smiles = train_smiles[:1000]
	# train_logDs = train_logDs[:1000]
	# test_smiles = test_smiles[:500]
	# test_logDs = test_logDs[:500]
	all_smiles = construct_graphs(train_smiles, train_logDs)
	all_test_smiles = construct_graphs(test_smiles, test_logDs)

	features = all_smiles[0][0].node_features

	if args.model == 'GCN_GAT':
		model = GCN_GAT(features.shape[1], hidden, 2)
	elif args.model == 'GCN':
		model = GCN(features.shape[1], hidden, 2)
	elif args.model == 'GAT' :
		model = GAT(features.shape[1], hidden, 2 )
	else: ## for baseline comparison
		model = GraphMLP(features.shape[1], 2)

	train_epoch_loss, epoch_acc, epoch_auc  = train_and_evaluation(args, iterations, model, all_smiles, train_logDs, batch_size, eval_interval, all_test_smiles, test_logDs)
	print ("All AUC", epoch_auc)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="GAT", help='GAT, GCN_GAT, GCN, MLP')	
	parser.add_argument("--iterations", type=int, default=1000)	
	parser.add_argument("--eval_interval", type=int, default=5)	
	parser.add_argument("--batch_size", type=int, default=32)	
	parser.add_argument("--active_index", type=bool, default=False)	
	args = parser.parse_args()

	main(args)