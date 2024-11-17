#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:32:53 2024

@author: jyzhang
"""

import networkx as nx
import torch
from torch_geometric.data import Data

import config as cfg
 
def generate_graph(num_nodes):
    m = 2    # Number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(num_nodes, m)

    # edge_prob = 0.2
    # G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    # nx.draw(G, with_labels=True)
    
    G = G.to_directed()

    edge_index = torch.tensor(list(G.edges), device=cfg.device).t().contiguous()
    edge_weight = torch.ones(edge_index.size(1), device=cfg.device)  # All edge weights are 1
    
    feat = torch.zeros((num_nodes, cfg.dim_in), dtype=torch.long, device=cfg.device)  # Feature matrix with zeros
    data = Data(feat=feat, edge_index=edge_index, edge_weight=edge_weight)
    
    return data
