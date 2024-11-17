# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:41:51 2024

@author: jyzhang
"""


import torch
import config as cfg


def calculate_cut_value(data, state):

    edge_index = data.edge_index
    edge_weight = data.edge_weight

    # Get the features for the source and target nodes of each edge
    src_feat = state[edge_index[0]]
    tgt_feat = state[edge_index[1]]

    # Calculate the cut value
    cut_value = edge_weight[(src_feat != tgt_feat)].sum().item()
    
    return cut_value


class Env:
    def __init__(self):
        pass


    def reset(self, data):
        self.data = data 
        self.max_steps = self.data.num_nodes-1
        
        self.state = torch.zeros(self.data.num_nodes, dtype=torch.long, device=cfg.device)
        # self.all_nodes = set(range(self.data.num_nodes))  # Create a set containing all nodes in the graph
        # self.partial_solution = set()
        
        self.cost_function = 0

        self.current_step = 0
        
        return self.state.clone()
  

    def step(self, action):
        action = action.item()
        if self.state[action] == 1:
            raise ValueError("Node has already been choosen")
        
        self.current_step += 1
        
        self.state[action] = 1

        cost_function_old = self.cost_function
        self.cost_function = calculate_cut_value(self.data, self.state)
        reward = self.cost_function - cost_function_old

        terminated = self.current_step >= self.max_steps
        # terminated = self.current_step >= 2
        truncated = reward < 0
        # print(self.data)
        
        # if truncated:
        #     reward = 0
        
        state_next = self.state.clone()
        return state_next, reward, terminated, truncated, {}
        


