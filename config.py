# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:14:16 2024

@author: jyzhang
"""


import torch


dim_in = 1
dim_embed = 32

lr = 1e-3
gamma = 1
tau = 0.05
buffer_size = 10000
batch_size = 16

epsilon_s = 0.95
epsilon_e = 0.05

# epsilon_decay = 10000

num_episodes = 10000

T = 3


num_nodes_lower = 30
num_nodes_upper = 50

C_lower = 0.1
C_upper = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


