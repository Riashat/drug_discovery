import csv
import numpy as np
import os
from matplotlib import pyplot as plt
#@title Importing RDKit for molecule parsing
import rdkit as rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
# from torch_geometric.nn import GCNConv, GATv2Conv
import math
import torch
import torch.nn as nn
import random
from pathlib import Path
from warnings import filterwarnings
import time
import pandas as pd
import numpy as np
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import auc, accuracy_score, recall_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
# Import required libraries for data analysis and visualisation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit as rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from torch.utils.data import Dataset
import logging
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances

# Data Loader
import pickle
import os

# Scaffold splitting
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress
from collections import defaultdict

# Utils
from sklearn.metrics import roc_auc_score

# # Define the attention model
import torch
import torch.optim as optim
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, clone
from sklearn.neural_network import MLPClassifier

from utils import one_hot_vector, pad_array


def featurize_mol(mol):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    conf = mol.GetConformer()
    node_features = np.array([get_atom_features(atom)
                            for atom in mol.GetAtoms()])
    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
    
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                        for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)
    
    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom):
    """Calculate atom features. 

    Args:
        atom (rdchem.Atom): An RDKit Atom object.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    attributes.append(atom.GetFormalCharge())
    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


class Molecule:

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index

class MolDataset(Dataset):

    def __init__(self, data_list):

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == list:
            return MolDataset([self.data_list[int(i)] for i in key])
        
        return self.data_list[key]



def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list = [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))


    return [torch.FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels)]
