import blackhc.project.script
from tqdm.auto import tqdm
import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms     

def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)

def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.

    """

    # convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    else:
        # NBVAL_CHECK_OUTPUT
        print(f"Warning: Wrong method specified: {method}. Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))

class BayesianNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, features, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(167, 64)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(64, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = self.fc1(input)
        input = self.fc1_drop(input)
        input = F.relu(input)
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input

def load_dataset():
    SEED = 0        
    data = 	pd.read_csv("/Users/Riashat/Documents/interviews/valence/drug_discovery/egfr_compounds_classification.csv",
        index_col=0,
    )
    # NBVAL_CHECK_OUTPUT
    print("Number of active compounds:", int(data.active.sum()))
    print("Number of inactive compounds:", len(data) - int(data.active.sum()))

    data_df = data.copy()
    data_df["fp"] = data_df["smiles"].apply(smiles_to_fp)

    fingerprint_to_model = data_df.fp.tolist()
    label_to_model = data_df.active.tolist()

    (
        static_train_x,
        static_test_x,
        static_train_y,
        static_test_y,
    ) = train_test_split(fingerprint_to_model, label_to_model, test_size=0.4, random_state=SEED)
    splits = [static_train_x, static_test_x, static_train_y, static_test_y]

    return splits, fingerprint_to_model, label_to_model

splits, fingerprint_to_model, label_to_model = load_dataset()

train_dataset_x = splits[0]
train_dataset_y = splits[2]
test_dataset_x = splits[1]
test_dataset_y = splits[3]

# num_initial_samples = 20
num_classes = 2


initial_samples = active_learning.get_balanced_sample_indices(
    train_dataset_y , num_classes=2, n_per_digit=10
)


## experiment
max_training_samples = 1800
acquisition_batch_size = 20
num_inference_samples = 100
num_test_inference_samples = 2
num_samples = 100
test_batch_size = 32
batch_size = 64
scoring_batch_size = 64
training_iterations = 4096 * 6

use_cuda = torch.cuda.is_available()

print(f"use_cuda: {use_cuda}")

device = "cuda" if use_cuda else "cpu"

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(list(zip(train_dataset_x, train_dataset_y)), batch_size=test_batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(list(zip(test_dataset_x, test_dataset_y)), batch_size=test_batch_size, shuffle=False, **kwargs)

# active_learning_data = active_learning.ActiveLearningData(train_dataset)
active_learning_data = active_learning.ActiveLearningData(train_loader.dataset)

# Split off the initial samples first.
active_learning_data.acquire(initial_samples)

# THIS REMOVES MOST OF THE POOL DATA. UNCOMMENT THIS TO TAKE ALL UNLABELLED DATA INTO ACCOUNT!
active_learning_data.extract_dataset_from_pool(500)

## trial
# active_learning_data.extract_dataset_from_pool(200)



train_loader = torch.utils.data.DataLoader(
    active_learning_data.training_dataset,
    sampler=active_learning.RandomFixedLengthSampler(active_learning_data.training_dataset, training_iterations),
    batch_size=batch_size,
    **kwargs,
)


pool_loader = torch.utils.data.DataLoader(
    active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs
)


# Run experiment
test_accs = []
test_loss = []
added_indices = []

pbar = tqdm(initial=len(active_learning_data.training_dataset), total=max_training_samples, desc="Training Set Size")

all_dataset_indices = []

while True:
    model = BayesianNN(features=167, num_classes=2).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    # Train
    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device=device)
        target = target.to(device=device)
        optimizer.zero_grad()

        data = data.to(torch.float32)
        prediction = model(data, 1).squeeze(1)
        
        loss = F.nll_loss(prediction, target.long())

        loss.backward()
        optimizer.step()

    # Test
    loss = 0
    correct = 0
    all_y_pred = []
    all_y_true = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data = data.to(device=device)
            target = target.to(device=device)

            
            data = data.to(torch.float32)
            prediction = torch.logsumexp(model(data, num_test_inference_samples), dim=1) 
            
            loss += F.nll_loss(prediction, target.long(), reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            ## appending y_pred and y_target
            all_y_pred.append(prediction.data.numpy())
            all_y_true.append(target.data.numpy())


    loss /= len(test_loader.dataset)
    test_loss.append(loss)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)
    test_accs.append(percentage_correct)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    epoch_roc_auc = roc_auc_score(y_true, y_pred)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )
    print("AUC Score", epoch_roc_auc)

    if len(active_learning_data.training_dataset) >= max_training_samples:
        break

    # Acquire pool predictions
    N = len(active_learning_data.pool_dataset)
    M = len(active_learning_data.training_dataset)
    logits_N_K_C = torch.empty((N, num_inference_samples, num_classes), dtype=torch.double, pin_memory=use_cuda)
    
    print ("Dataset Length")
    print("Pool Set", N)
    print("Train Set", M)

    with torch.no_grad():
        model.eval()

        for i, (data, _) in enumerate(tqdm(pool_loader, desc="Evaluating Acquisition Set", leave=False)):
            data = data.to(device=device)
            data = data.to(torch.float32)
            lower = i * pool_loader.batch_size
            upper = min(lower + pool_loader.batch_size, N)
            logits_N_K_C[lower:upper].copy_(model(data, num_inference_samples).double(), non_blocking=True)

    with torch.no_grad():
        candidate_batch = batchbald.get_batchbald_batch(
            logits_N_K_C, acquisition_batch_size, num_samples, dtype=torch.double, device=device
        )

    all_targets = []
    for i in range(len(active_learning_data.pool_dataset.indices)):
        x = train_dataset_y[i]
        all_targets.append(x)
    
    targets = all_targets

    # targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
    dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)
    all_dataset_indices.append(dataset_indices)
    # print("Dataset indices: ", dataset_indices)
    # print("Scores: ", candidate_batch.scores)
    # print("Labels: ", targets[candidate_batch.indices])
    active_learning_data.acquire(candidate_batch.indices)
    added_indices.append(dataset_indices)
    pbar.update(len(dataset_indices))

## 
# hide
# experiment
test_accs
# hide
# experiment
test_loss


    