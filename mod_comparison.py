# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:55:01 2024

@author: jyzhang
"""

from torch_geometric.utils import to_networkx
import numpy as np
 

def max_deg(data, state):
    
    remaining_nodes = (state == 0).nonzero(as_tuple=True)[0].tolist()

    # Convert to NetworkX graph
    G = to_networkx(data, edge_attrs=['edge_weight'])
    
    # Calculate weighted degree for each node
    weighted_degree = dict(G.degree(weight='edge_weight'))

     
    degrees = [weighted_degree[node] for node in remaining_nodes]
    
    action = remaining_nodes[ np.argmax(degrees) ]
    
    return action


