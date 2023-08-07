# Drug Discovery Problem 

## Summary of Contributions

## Predicting Potency Value for Novel Compunds targeting EGFR

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
1. Attention based model for the prediction task
(Molecule Attention Transformer from the paper : https://arxiv.org/pdf/2002.08264.pdf)
2. Graph Attention Network 
3. Graph Convolutional Network
4. Graph Attention and Convolutional Network
5. Deep Bayesian Neural Network with an Active Learning pipeline

##### Featurization - TODO
For ** Model 1 (Attention) ** we follow --- disscuss
For Others, we assume a graph structure 


## Description of Model and Approach, and Experimental Results

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



### Graph Attention Networks


### Graph Convolutional Network
We used an off the shelf GCN model implementation, and adapted the featurization scheme 
to use a MGCF featurizer from RDKIT to convert smiles into graph data. Performance of a GCN 
model is as given beloe. The model uses as input the graph node features and edge indices. 

----- Imbalanced Dataset : 


----- Balanced Dataset :

### Graph Attention Network
Similar to the attention mechanism used in the Molecule Transformer, we used attention in graph networks as 
well, to implementa Graph Attention Network (GAT). As before, the featurization is achieved with a MGCF featurizer
to convert smiles data into graph structure. The model uses as input the graph node features and edge indices. 


----- Imbalanced Dataset : 


----- Balanced Dataset :


### Graph Convolutional Network with Attention 

----- Imbalanced Dataset : 


----- Balanced Dataset :


## Bayesian Active Learning for Class Imbalance Problem









## Experimental Procedure and Discussion


## Experiment Results


## Failure Modes and Limitations
Structure mapping with active and inactive
Graph structure not there - so featurizers not useful that converts smiles to graph structure


## Future Work and Things Remaining To Do
