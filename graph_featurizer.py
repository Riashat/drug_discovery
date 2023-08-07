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

def construct_adjacency_matrix(node_features, edge_index):
	adj_mat = np.zeros([node_features.shape[0],node_features.shape[0]])

	for col_e in range(edge_index.shape[1]):
		source_at = int(edge_index[0,col_e])
		dest_at = int(edge_index[1,col_e])
		adj_mat[source_at,dest_at]=1

	return adj_mat


def construct_graphs(train_smiles, train_logDs):
	all_smiles = []
	# all_node_features = []
	# all_edge_index = []
	# all_adj_mat = []
	print("Constructing Node Features and Edge Adjacency")
	for idx, smiles in enumerate(train_smiles): 
		if idx == 3299:
			continue
		feat_graphs = MGCF(use_edges=True).featurize([smiles])[0]
		# all_node_features.append(feat_graphs.node_features)
		# all_edge_index.append(feat_graphs.edge_index)
		const_adj_mat = construct_adjacency_matrix(feat_graphs.node_features, feat_graphs.edge_index)
		# all_adj_mat.append(const_adj_mat)
		all_smiles.append((feat_graphs, const_adj_mat, train_logDs[idx]))

	# node_features = np.concatenate(all_node_features, axis=0)
	# edge_index = np.concatenate(all_edge_index, axis=1)

	return all_smiles

