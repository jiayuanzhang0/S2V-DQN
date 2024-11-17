# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:43:25 2024

@author: jyzhang
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.optim as optim

from collections import deque
import random

import config as cfg


def get_state_embed(embed):
    return torch.sum(embed, dim=0).unsqueeze(0)



class S2V(MessagePassing):
    def __init__(self, dim_in, dim_embed):
        super(S2V, self).__init__(aggr='add')
        self.theta1 = nn.Parameter(torch.Tensor(dim_in, dim_embed))
        self.theta2 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))
        self.theta3 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))
        self.theta4 = nn.Parameter(torch.Tensor(1, dim_embed))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)
        nn.init.xavier_uniform_(self.theta3)
        nn.init.xavier_uniform_(self.theta4)
    
    def forward(self, feat, edge_index, edge_weight, embed):
        message1 = self.propagate(edge_index, x=embed, edge_weight=edge_weight, message_type = '1')
        message2 = self.propagate(edge_index, x=embed, edge_weight=edge_weight, message_type = '2')
        out = F.relu( torch.matmul(feat,     self.theta1)  
                    + torch.matmul(message1, self.theta2) 
                    + torch.matmul(message2, self.theta3)   
                    )
        return out
    
    def message(self, x_j, edge_weight, message_type):
        # edge_weight is 1D vector. shape = batch size
        # x_j is 2D matrix. shape = batch size * dim_embed
        if message_type == '1':
            # print(x_j)
            return x_j
        
        elif message_type == '2':
            return F.relu(torch.matmul(edge_weight.unsqueeze(1), self.theta4))

    def update(self, aggr_out):
        return aggr_out


# input a batch of states and corresponding actions; output a batch of q
class QFunction(nn.Module):
    def __init__(self, dim_embed):
        super(QFunction, self).__init__()
        self.theta5 = nn.Parameter(torch.Tensor(2 * dim_embed, 1))
        self.theta6 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))
        self.theta7 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta5)
        nn.init.xavier_uniform_(self.theta6)
        nn.init.xavier_uniform_(self.theta7)

    def forward(self, state_embed, embed):
        
        theta7_muv = torch.matmul(embed, self.theta7)
        
        if state_embed.size(0) == 1:
            # one state and many actions. 
            theta6_state = torch.matmul(state_embed, self.theta6).expand(embed.size(0), -1)
        else:
            theta6_state = torch.matmul(state_embed, self.theta6)

        concat = torch.cat((theta6_state, theta7_muv), dim=1)
        q = torch.matmul(F.relu(concat), self.theta5)
        return q



class GSet():
    def __init__(self):
        self.g_list = deque([])
        
    def push(self, data):
        self.g_list.append(data)
        

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def push(self, gid, state, action, reward, state_next, done):
        self.buffer.append((gid, state, action, reward, state_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        gid, state, action, reward, state_next, done = zip(*batch)
        # vars: (array(batch), array(batch))
        
        gid = torch.cat(gid)
        # state = torch.cat(state)
        action = torch.cat(action)
        reward = torch.cat(reward)
        # state_next = torch.cat(state_next)
        done = torch.cat(done)

        return gid, state, action, reward, state_next, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent():
    
    def __init__(self):
        
        self.s2v_policy = S2V(dim_in=cfg.dim_in, dim_embed=cfg.dim_embed).to(cfg.device)
        self.s2v_target = S2V(dim_in=cfg.dim_in, dim_embed=cfg.dim_embed).to(cfg.device)
        
        self.qfunc_policy = QFunction(dim_embed=cfg.dim_embed).to(cfg.device)
        self.qfunc_target = QFunction(dim_embed=cfg.dim_embed).to(cfg.device)
        
        self.s2v_target.load_state_dict(self.s2v_policy.state_dict())
        self.qfunc_target.load_state_dict(self.qfunc_policy.state_dict())
        
        # Set the target_net to no gradients
        for param in self.s2v_target.parameters():
            param.requires_grad = False
        for param in self.qfunc_target.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(list(self.s2v_policy.parameters()) + list(self.qfunc_policy.parameters()), lr=cfg.lr)

        self.GSet = GSet()
        self.ReplayBuffer = ReplayBuffer(cfg.buffer_size)
        

    def encoding(self, data, encoding_type):
        # init embed
        embed = torch.zeros((data.num_nodes, cfg.dim_embed), device=cfg.device)  # Feature matrix with zeros
        
        if encoding_type == 'policy':
            for i in range(cfg.T):
                embed = self.s2v_policy(data.feat, data.edge_index, data.edge_weight, embed)
        elif encoding_type == 'target':
            for i in range(cfg.T):
                embed = self.s2v_target(data.feat, data.edge_index, data.edge_weight, embed)

        return embed
        
        
    def select_action(self, state_embed, embed, epsilon, state):
        # print('select action')
        random_num = random.random()
        remaining_nodes = (state == 0).nonzero(as_tuple=True)[0].tolist()
        
        # exploitation
        if random_num > epsilon:
            with torch.no_grad():
                qs = self.qfunc_policy(state_embed, embed)

                action = remaining_nodes[ torch.argmax(qs[remaining_nodes]).item() ]

                action = torch.tensor([[action]], dtype=torch.long, device=cfg.device)  
                
            # print('exploitation=====', action)
            return action
        # exploration
        else:
            action = torch.tensor([[random.choice(remaining_nodes)]], device=cfg.device, dtype=torch.long)
            # print('rand-------------', action)
            return action
    
    
    
    def train(self):
        # print('train')
        if len(self.ReplayBuffer) < cfg.batch_size:
            # print(len(self.buffer))
            return
        gid, state, action, reward, state_next, done = self.ReplayBuffer.sample(cfg.batch_size)
        
        
        data_list = [self.GSet.g_list[gid[i, 0]].clone() for i in range(cfg.batch_size)]
        for i, data in enumerate(data_list):
            data.feat = state[i].unsqueeze(1).float()
        data_batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        embed = self.encoding(data_batch, 'policy')  # embed will have embeddings for all nodes in all graphs
        state_embed = torch_geometric.utils.scatter(embed, data_batch.batch, dim=0, reduce='sum')
        
        action_indices = [action[i, 0].item() + data_batch.ptr[i].item() for i in range(cfg.batch_size)]
        action_embed = embed[action_indices]
        

        # state_embed and action_embed are both in batch
        q = self.qfunc_policy(state_embed, action_embed)
        # print(q)
        
        
        # TD target
        #----------------------------------------------------------------------
        with torch.no_grad():
            # print('target')
            
            
            state_next_cat = torch.cat(state_next, dim=0)  # Concatenates along the first dimension (batch dimension)
            remaining_node_indices = (state_next_cat == 0).nonzero(as_tuple=False).squeeze()
            
            
            data_list = [self.GSet.g_list[gid[i, 0]].clone() for i in range(cfg.batch_size)]
            for i, data in enumerate(data_list):
                data.feat = state_next[i].unsqueeze(1).float()
            data_batch = torch_geometric.data.Batch.from_data_list(data_list)
            
            # DQN
            #------------------------------------------------------------------
            embed = self.encoding(data_batch, 'policy')  # embed will have embeddings for all nodes in all graphs
            state_next_embed = torch_geometric.utils.scatter(embed, data_batch.batch, dim=0, reduce='sum')
            
            Q_next = self.qfunc_policy(state_next_embed[data_batch.batch[remaining_node_indices]], 
                                       embed[remaining_node_indices])
            #------------------------------------------------------------------
            
            
            action_star_indices = torch.zeros(cfg.batch_size, device=cfg.device).long()
            for i in range(cfg.batch_size):
                batch_mask_for_remaining_nodes = data_batch.batch[remaining_node_indices] == i
                
                action_index = batch_mask_for_remaining_nodes.nonzero(as_tuple=False).squeeze().min().item() + torch.argmax(Q_next[batch_mask_for_remaining_nodes]).item()
                action_star_indices[i] = remaining_node_indices[action_index]
            
            
            # target network
            #------------------------------------------------------------------
            embed = self.encoding(data_batch, 'target')  
            state_next_embed = torch_geometric.utils.scatter(embed, data_batch.batch, dim=0, reduce='sum')
            
            action_embed = embed[action_star_indices]            
            
            q_next = self.qfunc_target(state_next_embed, action_embed)
            #------------------------------------------------------------------
            
            q_target = reward + cfg.gamma * q_next * (1-done)
        #----------------------------------------------------------------------
        
        
        # update
        #----------------------------------------------------------------------
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()

        loss = criterion(q, q_target)
        # print('------------------')
        # print(q[:4], q_target[:4])
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #----------------------------------------------------------------------
        

        # soft update the target network
        for param, target_param in zip(self.s2v_policy.parameters(), self.s2v_target.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
        for param, target_param in zip(self.qfunc_policy.parameters(), self.qfunc_target.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
        
        # q_max = torch.max( q.detach() )
        return 
        