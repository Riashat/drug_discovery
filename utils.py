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

def train_val_test(X_smiles, Y):

	# feature_path = 'data_features.pkl'
	# if not os.path.exists(feature_path): 
	#     X_all, Y_all = load_data_from_smiles(x_smiles=X_smiles, labels= Y, normalize_features=False)
	#     print('Saving molecule features transformed from raw smiles')
	#     pickle.dump((X_all, Y_all), open(feature_path, 'wb'))
	# else:
	#     X_all, Y_all = pickle.load(open(feature_path, "rb"))

	train_idx = [np.random.randint(len(X_smiles)) for _ in range(int(0.6*len(X_smiles)))]

	all_idx = [i for i in range(len(X_smiles))]
	val_idx = [i for i in all_idx if i not in train_idx]

	train_dataset = []
	train_y = []
	for i in train_idx:
		train_dataset.append(X_smiles[i])
		train_y.append(Y[i])
	
	val_dataset = []
	val_y = []
	for j in val_idx : 
		val_dataset.append(X_smiles[j])
		val_y.append(Y[j])

	return train_dataset, train_y, val_dataset, val_y

def load_data(active=False):

    if not active:
        data = 	pd.read_csv("/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds_classification.csv",
            index_col=0,
        )
        print("Number of active compounds:", int(data.active.sum()))
        print("Number of inactive compounds:", len(data) - int(data.active.sum()))

        #@title Load data / split to train and test
        with open('/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds_classification.csv') as fh:
            smiles = []
            logDs = []
            header = True
            for row in csv.reader(fh):
                if header:
                    header = False
                    continue
                smiles.append(row[2])  
                logDs.append(float(row[4]))

        train_dataset, train_y, val_dataset, val_y = train_val_test(smiles, logDs)

    else:
        data = 	pd.read_csv("/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds_classification.csv",
            index_col=0,
        )

        active_indices = np.load("active_indices.npy")
        data = data.iloc[active_indices, :]

        print("Number of active compounds:", int(data.active.sum()))
        print("Number of inactive compounds:", len(data) - int(data.active.sum()))

        #@title Load data / split to train and test
        with open('/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds_classification.csv') as fh:
            smiles = []
            logDs = []
            header = True
            for row in csv.reader(fh):
                if header:
                    header = False
                    continue
                smiles.append(row[2])  
                logDs.append(float(row[4]))

        train_dataset, train_y, val_dataset, val_y = train_val_test(smiles, logDs)
        
	
    return train_dataset, train_y, val_dataset, val_y



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc



def binary_acc2(y_pred, y_test):
    y_pred_tag = torch.round(torch.Tensor(y_pred))
    y_test = torch.Tensor(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc



class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def data_load():

    data = pd.read_csv("/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds.csv",
        index_col=0,
    )

    print("Shape of dataframe : ", data.shape)
    data.head()

    # Keep only the columns we want
    data = data[["molecule_chembl_id", "smiles", "pIC50"]]
    # data = data.iloc[:1800, :]

    data.head()
    # NBVAL_CHECK_OUTPUT

    data["active"] = np.zeros(len(data))

    # Mark every molecule as active with an pIC50 of >= 6.3, 0 otherwise
    data.loc[data[data.pIC50 > 8.0].index, "active"] = 1.0

    # NBVAL_CHECK_OUTPUT
    print("Number of active compounds:", int(data.active.sum()))
    print("Number of inactive compounds:", len(data) - int(data.active.sum()))
    # Number of active compounds: 873
    # Number of inactive compounds: 3762

    data.head()

    data.to_csv("egfr_compounds_classification.csv")
        
    return data


def draw_smiles(data):

	# Get SMILES string
	smiles = data["smiles"][0]
	mol = Chem.MolFromSmiles(smiles) 

	print(f"SMILES: {smiles}, Number of atoms: {len(mol.GetAtoms())}")
	Draw.MolToImage(mol)



def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    min_vec, max_vec = x_all[0][0].min(axis=0), x_all[0][0].max(axis=0)
    for x in x_all:
        min_vec = np.minimum(min_vec, x[0].min(axis=0))
        max_vec = np.maximum(max_vec, x[0].max(axis=0))
    diff = max_vec - min_vec
    diff[diff == 0] = 1.

    for x in x_all:
        afm = x[0]
        afm = (afm - min_vec) / diff
        x[0] = afm

    return x_all



def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def random_scaffold_split(smiles_list, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param smiles_list: list of smiles corresponding to the dataset obj
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    N = len(smiles_list)
    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * N))
    n_total_test = int(np.floor(frac_test * N))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return train_idx, valid_idx, test_idx



def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False





