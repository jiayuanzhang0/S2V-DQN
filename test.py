# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:35:38 2024

@author: jyzhang
"""

import os
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp

import mod_agent
import maxcut_env
import mod_comparison
from mod_agent import DQNAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved DQN model
with open('./dqn.pkl', 'rb') as f:
    agent = pickle.load(f)

def load_graphs(data_dir):
    graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    graphs = [torch.load(os.path.join(data_dir, graph_file)) for graph_file in graph_files]
    for f in graph_files:
        print(f'Graph {f} loaded.')
    return graphs

test_data_dir = './data/'  
test_graphs = load_graphs(test_data_dir)

env = maxcut_env.Env()

# Helper function to run a single policy
def run_policy(data, policy_fn, label, color, epsilon=0):
    state = env.reset(data.clone())
    reward_episode = 0
    reward_episode_list = [0]

    while True:
        action = policy_fn(data, state, epsilon)
        state_next, reward, terminated, truncated, _ = env.step(action)
        reward_episode += reward
        reward_episode_list.append(reward_episode)

        if terminated or truncated:
            break
        state = state_next.clone()

    plt.plot(reward_episode_list[:-1], color=color, label=label)

# Policy functions
def random_policy(data, state, epsilon):
    data.feat = state.unsqueeze(1).float()
    embed = agent.encoding(data, 'policy')
    state_embed = mod_agent.get_state_embed(embed)
    return agent.select_action(state_embed, embed, epsilon, state)

def max_degree_policy(data, state, epsilon):
    action = mod_comparison.max_deg(data, state)
    return torch.tensor([[action]], dtype=torch.long)

def goemans_williamson_policy(data):
    edges = data.edge_index.t().tolist()
    X = cp.Variable((data.num_nodes, data.num_nodes), symmetric=True)
    constraints = [X >> 0] + [X[i, i] == 1 for i in range(data.num_nodes)]
    objective = sum(0.5 * (1 - X[i, j]) for i, j in edges)
    cp.Problem(cp.Maximize(objective), constraints).solve()

    x = sqrtm(X.value)
    u = np.random.randn(data.num_nodes)  # Random hyperplane normal
    x = np.sign(x @ u)

    # Convert x to state tensor
    state = torch.zeros(len(x), dtype=torch.long, device=device)
    state[x.real == -1] = 0
    state[x.real == 1] = 1
    return state

# Plot setup
plt.figure(figsize=(10, 6))
fontsize = 20

# Run policies
data = test_graphs[0]
run_policy(data, random_policy, 'Random', 'blue', epsilon=1)
run_policy(data, max_degree_policy, 'Max Degree', 'orange')
run_policy(data, random_policy, 'DQN', 'red', epsilon=0)

# Goemans-Williamson Solution
gw_state = goemans_williamson_policy(data)
gw_cost = maxcut_env.calculate_cut_value(data, gw_state)
plt.axhline(y=gw_cost, color='green', label='Goemans-Williamson')
plt.axhline(y=gw_cost / 0.87, color='black', label='GW/0.87')
print(f"GW cost: {gw_cost}")
print(f"GW cost / 0.87: {gw_cost / 0.87}")

# Random Flip Policy
random_state = env.reset(data.clone())
flip_indices = torch.randperm(random_state.size(0))[:random_state.size(0) // 2]
random_state[flip_indices] = 1
random_cost = maxcut_env.calculate_cut_value(data, random_state)
plt.axhline(y=random_cost, color='purple', label='Random Flip')


plt.xlabel('Step', fontsize=fontsize)
plt.ylabel('Return', fontsize=fontsize)
plt.legend(fontsize=fontsize * 0.7)
plt.grid(True)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('./step_return.png')




