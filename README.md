## Predicting Potency Value for Novel Compunds targeting EGFR

## Summary of Contributions

We highlight the main contributions first (with details below):

a. *Class Imbalance Problem* : The dataset has a class imbalance problem; we follow a deep bayesian active learning 
pipeline to handle imbalanced dataset

b. *Several Models for Comparison, with different featurization from smiles representation* : We train and compare several models, each with different featurization from the smiles representation. The list of 
models we implemented and compared performance with are : 
(i) Bayesian NN classifier based on a RDKit Molecule Featurization
(ii) Molecule attention transformer [1] is implemented, based on MolFromSmiles featurization, where input data 
is a list of graph descriptors (node features, adjacency matrices, distance matrices)
(iii) Graph Attention Network based on MGCF Featurization, assuming data has a graph structure mapping to active and inactive compounds
(iv) Graph Convolutional Netork based on MGCF Featurization
(v) Combining (iv) and (iii) for a Graph Convolutional and Attention Network

For (b), we assumed the dataset has a graph molecule structure, and the above models (mostly (ii to v)) should be able to
better capture the graph structure for classification of active compounds. 

c. *Active Learning Pipeline for Small Data Regime* : Finally, even though the dataset has 4.5k compounds, it has a class 
imbalance problem and the number of active compounds (pIC50 >8) is relatively small (only). This means, if we take an equal class distribution of active and inactive compounds, the total datapoint would be around ~2000 compounds only (half the original dataset). To handle class imbalance and have more informative datapoints to train from, we follow a *Deep Bayesian Active Learning* pipeline, where we train a Bayesian NN with Monte-Carlo Dropout (Gal et al., 2016) and query batches of uncertain, but informative points from the poolset, based on the Deep BALD acquisition function. We then compare performance of our models in (b) based on the imbalanced and balanced datasets. 

d. Experimental Results evaluate the classification problem on EGFR dataset, and we report ROC/AUC value and test/validation set accuracy of predictions. All models are trained on CPU and not trained for long episodes/iterations, without any hyper-parameter fine-tuning. We believe performance of these models can be significantly improved with careful training and fine-tuning, and we can acheive better results. 

*Comments* : Experimental results comparing different models show that performance can vary significantly depending on the type of featurization being used to get molecule features from the smiles representation. For e.g when we used graph networks with MGCF featurization, performance can drop significantly, compared to using more standard featurizations for this dataset (e.g rdFingerprintGenerator) and training a standard NN classifier without any graph network. This shows that if proper domain knowledge was available, the choice of featurization to use indicates a lot about the type of model ideal for use in this dataset. Our results can be improved further if we could try and pick the best type of featurizations (could not do it due to lack of compute resources and time). Nevertheless, this repository should give an expeirmental comparison of different models for the EGFR dataset, and provides an active learning pipeline to improve the ratio of active and inactive compounds (and improve on the class balancing problem). 

## Approach

We consider the problem of predicting pIC50 potency values, as a classification problem, 
where active compounds with pIC50 > 8 are assigned a label 1 and all other compunds with pIC50 less than
or equal to 8 are assigned a label 0. 

The **key problem** when considering the dataset with 4.6k compunds is the * Class Imbalance * problem, 
where :
Number of active compounds: 873
Number of inactive compounds: 3762

Below we discuss the decision making process we took to detect active compunds, and the models that have been tried for this. 
In particular, we try *a few models* for this dataset, under different molecular featurizations, and take an *active learning* route to obtain a balanced dataset consisting of only the informative samples. One key challenge for this dataset is that since the number of active compounds is few, we work on a small data regime, where the small or active dataset is selected by following a *Deep Bayesian Active Learning* pipeline, as discussed below. 

### List of Models for classification on this dataset 
1. Deep Bayesian Neural Network with an Active Learning pipeline 
2. Attention based model for the prediction task (Molecule Attention Transformer)
3. Graph Attention Network 
4. Graph Convolutional Network
5. Graph Attention and Convolutional Network


### Featurization 
We use several different types of featurization, depending on the model at use. For e,g for the Graph Convolutional and Attention networks (GCN and GAT) we use a MGCF featurizer to conver the smiles representation, to get GraphData with node features, edge indices and adjacency matrices. Similar attributes can also be obtained, if we use other types of featurizatioon (e.g to get molecule representations from smiles), when we used a Marked Attention Transformer (MAT). Other than graph networks and attention transformers, we also trained a relatively simple Bayesian NN with Monte-Carlo dropout, where we used a rdFingerprintGenerator based featurization of the smiles representation to train a standard Bayesian classifier on this dataset. Performance varies significantly depending on the featurization. 


## Description of Model and Approach, with Experimental Results

### Deep Bayesian Active Learning for Imbalanced and Samll Data Regime
```
python main_active_learning.py
```
Due to the class imbalance problem in this EGFR dataset, we first follow an active learning pipeline to query most informative datapoints, and also be able to train with balanced and small data regime. To do this, we use a Bayesian NN with Monte-Carlo dropout as a classifier to train on small samples, while querying informative samples from a pool set, based on the BALD (Bayesian Active Learning by Disagreement) acquisition function. Experiment results show that as we start with small number of samples, and query and add datapoints to the training set, where the training set now contains a better ratio of active to non-acrive compounds, then performance based on the ROC/AUC metric improves; we use a standard featurization for this, assuming no graph structure in the data. 

From active learning, we query datapoints to eventually train with a total of 1800 training samples. This is because the original dataset was imbalanced with 873 active compounds - so we start from 20 training samples with equal class labels, and query points from the poolset, for a total of ~873 x 2 to reach a more balanced class ratio dataset.

#### Experimental Results with Active Learning from Small Data 

Results show how performance improves as start with only 20 training samples of active and inactive compounds, and query points from pool set to reach a larger number of informative training set
```
Training Set Size:  17%|██████████                                                  | 300/1800 [01:57<09:05,  2.75it/sTest set: Average loss: -0.0782, Accuracy: 1450/1854 (78.21%)                                                           
AUC Score 0.5685835976104433                                                                                           
Dataset Length
Pool Set 1981
Train Set 300
Training Set Size:  18%|██████████▋                                                 | 320/1800 [02:02<08:17,  2.98it/sTest set: Average loss: -0.0881, Accuracy: 1465/1854 (79.02%)                                                           
AUC Score 0.5891879932148388 
Dataset Length
Pool Set 1961
Train Set 320
Training Set Size:  19%|███████████▎                                                | 340/1800 [02:07<07:39,  3.18it/sTest set: Average loss: -0.1330, Accuracy: 1464/1854 (78.96%)                                                           
AUC Score 0.5878143668412125                                                                                           
Dataset Length
Pool Set 1941
Train Set 340
Training Set Size:  20%|████████████                                                | 360/1800 [02:13<07:13,  3.32it/sTest set: Average loss: -0.1388, Accuracy: 1484/1854 (80.04%)                                                           
AUC Score 0.5831071612950808                                                                                           
Dataset Length
Pool Set 1921
Train Set 360
Training Set Size:  21%|████████████▋                                               | 380/1800 [02:18<06:46,  3.49it/sTest set: Average loss: -0.1442, Accuracy: 1482/1854 (79.94%)                                                           
AUC Score 0.5886643557784498                                                                                           
Dataset Length
Pool Set 1901
Train Set 380
Training Set Size:  22%|█████████████▎                                              | 400/1800 [02:23<06:25,  3.63it/sTest set: Average loss: -0.1268, Accuracy: 1479/1854 (79.77%)                                                           
AUC Score 0.5855815325613983                                                                                           
Dataset Length
Pool Set 1881
Train Set 400
Training Set Size:  23%|██████████████                                              | 420/1800 [02:29<06:36,  3.48it/sTest set: Average loss: -0.1604, Accuracy: 1465/1854 (79.02%)                                                           
AUC Score 0.6078729994837377                                                                                           
Dataset Length
Pool Set 1861
Train Set 420
Training Set Size:  24%|██████████████▋                                             | 440/1800 [02:35<06:35,  3.44it/sTest set: Average loss: -0.1384, Accuracy: 1480/1854 (79.83%)                                                           
AUC Score 0.5952596061656464                                                                                           
Dataset Length
Pool Set 1841
Train Set 440
Training Set Size:  26%|███████████████▎                                            | 460/1800 [02:40<06:20,  3.52it/sTest set: Average loss: -0.1528, Accuracy: 1481/1854 (79.88%)                                                           
AUC Score 0.5862526735009957                                                                                           
Dataset Length
Pool Set 1821
Train Set 460
Training Set Size:  27%|████████████████                                            | 480/1800 [02:45<06:01,  3.65it/sTest set: Average loss: -0.1433, Accuracy: 1479/1854 (79.77%)                                                           
AUC Score 0.5772770853307766                                                                                           
Dataset Length
Pool Set 1801
Train Set 480
Training Set Size:  28%|████████████████▋                                           | 500/1800 [02:52<06:13,  3.48it/sTest set: Average loss: -0.1391, Accuracy: 1473/1854 (79.45%)                                                           
AUC Score 0.6115956191459547                                                                                           
Dataset Length
Pool Set 1781
Train Set 500
Training Set Size:  29%|█████████████████▎                                          | 520/1800 [02:59<06:46,  3.15it/sTest set: Average loss: -0.1479, Accuracy: 1484/1854 (80.04%)                                                           
AUC Score 0.596601888044841 
```

#### Achieving a Balanced Dataset from Active Learning : 
After running the active learning pipeline, we save the dataset indices of the most informative samples; these indices are then used to construct a new dataset, consisting of only the samples that were queried from active learning. This is denoted as the *Balanced Dataset* and we compare performance between the imbalanced and balanced datasets, for each of the models below


### Attention Transformer for Molecules
Firstly, we use a molecule attention transformer (MAT) since it has been shown to be useful 
for molecular graph structure. MAT can perform relatively well on several molecular prediction tasks 
and we chose this model for the EGFR prediction task. 

For this model, we feature the list of SMILES and prepare a list of 
graph descriptions consisting of node features, adjacency matrices describing the graph and 
distance matrix.

We implemented the MAT on this dataset, by using a train/validation split. The model takes quite some 
time to run on cpu, so we could not train it to full-extent. Below are results on both the imbalanced and the balanced dataset


----- Imbalanced Dataset : 
Epoch 008: | Train Loss: 0.49498 | Train Acc: 80.333| Train ROC: 0.570| 
                 Val Loss: 0.47767 | Val Acc: 81.333 | Val ROC: 0.593

Epoch 009: | Train Loss: 0.49505 | Train Acc: 80.000| Train ROC: 0.578| 
                 Val Loss: 0.47121 | Val Acc: 82.000 | Val ROC: 0.595


------ Balanced Dataset : 
TODO


### Graph Attention Networks

We trained a graph neural network with attention (GAT) by featurizing the smiles representation to obtain a graph structure from the data, consisting of node features and edge indices. Experimental results below shows performance when a GAT is used on both the balanced and imbalanced datasets. 

Active Index : False (Denotes using the full imbalanced data)

```
python main_gcn_gat.py --model GAT
```

Experimental results :

```
FULL DATASET (Imbalanced Classes)
Number of active compounds: 873
Number of inactive compounds: 3762

After initial training epochs; 
ROC values are low; can be improved further with careful model training

Eval Acc 0.8015594541910331
ROC AUC 0.5

```

Active Index : True (Denotes using a small but balanced dataset)
```
python main_gcn_gat.py --model GAT --active_index True
```

Experimental Results : 

```
Number of active compounds: 560
Number of inactive compounds: 1240

^^^ Better ratio of active to inactive compounds after the active learning pipeline

Eval Acc 0.8089097303634232
ROC AUC 0.5

```



### Graph Convolutional Network
We used an off the shelf GCN model implementation, and adapted the featurization scheme 
to use a MGCF featurizer from RDKIT to convert smiles into graph data. Performance of a GCN 
model is as given beloe. The model uses as input the graph node features and edge indices. 

----- Balanced Dataset :


```
python main_gcn_gat.py --model GCN --active_index True
```

```
Eval Acc 0.8093005080109418
ROC AUC 0.5
```


### Graph Convolutional Network with Attention 
```
python main_gcn_gat.py --model GCN_GAT --active_index True
```

----- Balanced Dataset :
```
Eval Acc 0.8158200853043815
ROC AUC 0.5

```



## Failure Modes and Limitations
Structure mapping with active and inactive
Graph structure not there - so featurizers not useful that converts smiles to graph structure


## Future Work and Things Remaining To Do




### References : 
[1] Molecule Attention Transformer (from the paper : https://arxiv.org/pdf/2002.08264.pdf)