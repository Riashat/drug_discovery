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



'''
---------

ATTENTION AND TRANSFORMER MODELS (Molecule Attention Transformer)

---------
'''

from torch import nn, einsum
from einops import rearrange

### Model Definition

# constants
DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim = -1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return d if not exists(val) else val

# helper classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, Lg = 0.5, Ld = 0.5, La = 1, dist_kernel_fn = 'exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # hyperparameters controlling the weighted linear combination from
        # self-attention (La)
        # adjacency graph (Lg)
        # pair-wise distance matrix (Ld)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask = None, adjacency_mat = None, distance_mat = None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h = h, qkv = 3).unbind(dim = -2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        assert dist_kernel_fn in DIST_KERNELS, f'distance kernel function needs to be one of {DISTANCE_KERNELS.keys()}'
        dist_kernel_config = DIST_KERNELS[dist_kernel_fn]

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(distance_mat):
                # mask distance to infinity
                # todo - make sure for softmax distance kernel, use -infinity
                dist_mask_value = dist_kernel_config['mask_value_fn'](dots)
                distance_mat.masked_fill_(~mask, dist_mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim = -1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        if exists(distance_mat):
            distance_mat = dist_kernel_config['fn'](distance_mat)
            attn = attn + Ld * distance_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class
class AttenModel(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        model_dim,
        dim_out,
        depth,
        heads = 8,
        Lg = 0.5,
        Ld = 0.5,
        La = 1,
        dist_kernel_fn = 'exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])
        self.attention = Attention(model_dim, heads = heads, Lg = Lg, Ld = Ld, La = La, dist_kernel_fn = dist_kernel_fn)

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, self.attention)),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
        mask = None,
        adjacency_mat = None,
        distance_mat = None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask = mask,
                adjacency_mat = adjacency_mat,
                distance_mat = distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim = -2)
        x = self.ff_out(x)
        x = self.sigmoid(x)
        return x



'''
---------

GRAPH NETWORK RELATED MODELS

---------
'''
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        # Weight
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
                        
    def forward(self, x, adj):
        batch_size = 1 
        x = x.reshape(batch_size, x.shape[0], x.shape[1])
        node_count = x.size()[1]
        x = x.reshape(batch_size * node_count, x.size()[2])
        x = torch.mm(x, self.weight)
        x = x.reshape(batch_size, node_count, self.weight.size()[-1])
        # Attention score
        attention_input = torch.cat([x.repeat(1, 1, node_count).view(batch_size, node_count * node_count, -1), x.repeat(1, node_count, 1)], dim=2).view(batch_size, node_count, -1, 2 * self.out_features)
        e = F.relu(torch.matmul(attention_input, self.weight2).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)        
        ## input 1 : bxnxm
        ## input 2 : bxmxp
        ##output : bxnxp
        x = torch.bmm(attention, x) 
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()

        self.gat1 = GraphAttentionLayer(dim_in, dim_h, dropout=0.1)
        self.gat2 = GraphAttentionLayer(dim_in, dim_h, dropout=0.1)
        self.fc1 = nn.Linear(128, 128)
        self.hc = torch.nn.Linear(128, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=0.005,
                                            weight_decay=5e-4)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.softmax
                
    def forward_attention(self, x, adj_mat):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, adj_mat)
        h = self.gat2(x, adj_mat)
        h = torch.max(h, dim=1)[0].squeeze() # max readout
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        x = F.relu(self.fc1(h))
        x = self.hc(x)
        x = x.reshape(1, x.shape[0])
        out = F.log_softmax(x, dim=1)
        softmax = torch.exp(out) 
        prob = list(softmax.detach().numpy())
        predictions = np.argmax(prob, axis=1)
        return x, out, predictions

    def forward(self, x, edge_index):
        x, out, pred = self.forward_attention(x, edge_index)
        return x, out, pred


class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.001,
                                      weight_decay=5e-4)
    self.sigmoid = torch.nn.Sigmoid()
    self.softmax = torch.softmax

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    out = F.log_softmax(h, dim=1)
    softmax = torch.exp(out) #output from linear
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)

    return h, out, predictions



class GCN_GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)
    self.sigmoid = torch.nn.Sigmoid()
    self.softmax = torch.softmax

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)

    out = F.log_softmax(h, dim=1)
    softmax = torch.exp(out) #output from linear
    prob = list(softmax.detach().numpy())
    predictions = np.argmax(prob, axis=1)
    
    return h, out, predictions



class GraphMLP(torch.nn.Module):
    def __init__(self, features, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.softmax = torch.softmax
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, features, edge_index):
        input = self.fc1(features)
        input = F.relu(input)
        input = self.fc2(input)
        out = F.log_softmax(input, dim=1)

        softmax = torch.exp(out) #output from linear
        prob = list(softmax.detach().numpy())
        predictions = np.argmax(prob, axis=1)
    
        return input, out, predictions
