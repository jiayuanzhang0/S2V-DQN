# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:05:15 2024

@author: jyzhang
"""


import os 
import torch

import graph 

       
save_dir = './data/'
num_graphs = 1
num_nodes=50
 
if not os.path.exists(save_dir): os.makedirs(save_dir)
    

for i in range(num_graphs):
    data = graph.generate_graph(num_nodes)
    torch.save(data, os.path.join(save_dir, 'graph_%04d.pt'%i))
    print(f'Graph {i} saved.')
    
        





