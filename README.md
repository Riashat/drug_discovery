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




##### References : 
[1] Molecule Attention Transformer (from the paper : https://arxiv.org/pdf/2002.08264.pdf)