import csv
import numpy as np
import os
#@title Importing RDKit for molecule parsing
import rdkit as rdkit
import rdkit.Chem as Chem
import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import random
from pathlib import Path
from warnings import filterwarnings
import time
import pandas as pd
import numpy as np
from rdkit import Chem
# Import required libraries for data analysis and visualisation
import pandas as pd
import rdkit as rdkit

# from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles

import logging
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles

# Data Loader
import pickle

# Utils
from sklearn.metrics import roc_auc_score
# # Define the attention model
import torch.optim as optim

from utils import DotDict, data_load, feature_normalize
from utils import  EarlyStopper, binary_acc
from molecule_featurizer import featurize_mol, get_atom_features, Molecule, MolDataset, mol_collate_func

from models import  AttenModel

def load_data_from_smiles(x_smiles, labels, normalize_features=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        normalize_features (bool): If True, Normalize features. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """

    x_all, y_all = [], []
    for smiles, label in zip(x_smiles, labels):
        try:

            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol)
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    if normalize_features:
        x_all = feature_normalize(x_all)
    return x_all, y_all


def train_val_test(X_smiles, Y):

    feature_path = 'data_features.pkl'
    if not os.path.exists(feature_path): 
        X_all, Y_all = load_data_from_smiles(x_smiles=X_smiles, labels= Y, normalize_features=False)
        print('Saving molecule features transformed from raw smiles')
        pickle.dump((X_all, Y_all), open(feature_path, 'wb'))
    else:
        X_all, Y_all = pickle.load(open(feature_path, "rb"))

    train_idx = [np.random.randint(len(X_smiles)) for _ in range(int(0.6*len(X_smiles)))]
    all_idx = [i for i in range(len(X_smiles))]
    val_idx = [i for i in all_idx if i not in train_idx]


    # Wrap data to Molecule object format and create dataset
    molecules = [Molecule(data[0], data[1], i) for i, data in enumerate(zip(X_all, Y_all))]
    dataset = MolDataset(molecules)

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = val_dataset


    return train_dataset, val_dataset, test_dataset, X_all[0][0].shape[1] 




def eval(args, model, device, loader, criterion=None):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            adjacency, features, distance, labels = batch
            adjacency, features, distance, labels = adjacency.to(device), features.to(device), distance.to(device), labels.to(device)
    
            pred = model(features, adjacency_mat = adjacency, distance_mat = distance)
            if criterion:
                loss = criterion(pred, labels)
                epoch_loss += loss.item()

            acc = binary_acc(pred, labels)
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

            epoch_acc += acc.item()

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        epoch_roc_auc = roc_auc_score(y_true, y_pred)

    if criterion:
        return epoch_loss/len(loader), epoch_acc/len(loader), epoch_roc_auc
    else:
        return epoch_acc/len(loader), epoch_roc_auc


def train(args, model, device, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    y_true = []
    y_pred = []
        
    for step, batch in enumerate(loader):
        adjacency, features, distance, labels = batch
        adjacency, features, distance, labels = adjacency.to(device), features.to(device), distance.to(device), labels.to(device)
        
        pred = model(features, adjacency_mat = adjacency, distance_mat = distance)
        
        flatten_adj =  adjacency.numpy().reshape(adjacency.shape[0], adjacency.shape[1] * adjacency.shape[2])   
        flatten_features =  features.numpy().reshape(features.shape[0], features.shape[1] * features.shape[2])   
        flatten_distance =  distance.numpy().reshape(distance.shape[0], distance.shape[1] * distance.shape[2])
        flatten_labels =  labels.numpy()   
        train_y = flatten_labels.reshape(-1,)   

        ## Loss matrix
        loss = criterion(pred, labels)

        with torch.no_grad():
            acc = binary_acc(pred, labels)
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    epoch_roc_auc = roc_auc_score(y_true, y_pred)

    return epoch_loss/len(loader), epoch_acc/len(loader), epoch_roc_auc


# Configs
def run_train(cfg, model, train_loader, val_loader, test_loader, device):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, nesterov=True, weight_decay=5e-04, momentum=0.9)
    if cfg.stepsize is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.stepsize, gamma=cfg.gamma)
    # Loss function
    criterion = nn.BCELoss() # TODO: some additional metric loss augmented with it such as center loss or robust loss

    early_stopper = EarlyStopper(patience=3, min_delta=1)
    best_val_roc = None

    for epoch in range(1, cfg.epochs+1):
        print("====epoch " + str(epoch))

        train_loss, train_acc, train_roc = train(cfg, model, device, train_loader, optimizer, criterion)
        if cfg.stepsize > 0: 
            scheduler.step()

        val_loss, val_acc, val_roc = eval(cfg, model, device, val_loader, criterion=criterion)
        test_acc = val_acc
        test_roc = val_roc

        print(f'Epoch {epoch+0:03}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}| Train ROC: {train_roc:.3f}| \n \
                Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.3f} | Val ROC: {val_roc:.3f}\n \
                Test Acc: {test_acc:.3f}| Test ROC: {test_roc:.3f}'
            )
        if early_stopper.early_stop(val_loss) or epoch == cfg.epochs:
            torch.save(
            {
                "epoch": epoch,
                "model_state_dict":  model.state_dict(),
            },
            "last.pt"
            )
            break

        if (best_val_roc == None or best_val_roc < val_roc):
            best_val_roc = val_roc
            torch.save(
            {
                "epoch": epoch,
                "model_state_dict":  model.state_dict(),
            },
            "best.pt"
            )


def main():

    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = data_load()
    cfg=DotDict({})

    cfg.lr = 0.001
    cfg.batch_size = 1080
    cfg.epochs= 50
    cfg.eval_train = True
    cfg.stepsize = 10
    cfg.gamma = 0.5

    X_smiles = data["smiles"].values
    Y = data["active"].values
    train_dataset, val_dataset, test_dataset, d_atom = train_val_test(X_smiles=X_smiles, Y=Y) ## dataset split
    batch_size = cfg.batch_size

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            collate_fn=mol_collate_func,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            collate_fn=mol_collate_func,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=mol_collate_func,
                                            shuffle=True)

    model = AttenModel(
        dim_in = d_atom,
        model_dim = 512,
        dim_out = 1,
        depth = 6,
        Lg = 0.5,                   # lambda (g)raph - weight for adjacency matrix
        Ld = 0.5,                   # lambda (d)istance - weight for distance matrix
        La = 1,                     # lambda (a)ttention - weight for usual self-attention
        dist_kernel_fn = 'exp'      # distance kernel fn - either 'exp' or 'softmax'
        )

    ## init model
    init_type = 'uniform'
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)

    device = torch.device("cpu") 

    run_train(cfg, model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader,
                device = device
                )   

    return data



if __name__ == "__main__":
	main()



