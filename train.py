# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:01:01 2024

@author: jyzhang
"""

import torch

import random
import pickle

import mod_agent
import maxcut_env
import config as cfg
import graph 


env = maxcut_env.Env()
agent = mod_agent.DQNAgent()


log_path = './log.txt'
with open(log_path, 'wb') as log_file:
    log_file.write(b'')  

# step_total = 0
for episode in range(cfg.num_episodes+1):
    print('episode', episode)
    
    reward_episode = 0
    
    num_nodes = random.randint(cfg.num_nodes_lower, cfg.num_nodes_upper)

    data = graph.generate_graph(num_nodes)
    state = env.reset(data.clone())
    agent.GSet.push(data.clone())
    gid = episode
    gid = torch.tensor([gid], dtype=torch.long, device=cfg.device).unsqueeze(0)
    
    # loop for one graph
    while True:
        # exponentially decay
        # epsilon = epsilon_e + (epsilon_s - epsilon_e) * math.exp(-1. * steps_total / epsilon_decay)
        
        # linearly decay
        epsilon = cfg.epsilon_s - (cfg.epsilon_s - cfg.epsilon_e) * (episode / cfg.num_episodes)

        data.feat = state.unsqueeze(1).float()
        embed = agent.encoding(data, 'policy')
        state_embed = mod_agent.get_state_embed(embed)
        
        action = agent.select_action(state_embed, embed, epsilon, state)
        state_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # state      = torch.tensor([state],         dtype=torch.long,    device=cfg.device).unsqueeze(0)
        reward     = torch.tensor([reward],        dtype=torch.float32, device=cfg.device).unsqueeze(0)
        # state_next = torch.tensor([state_next],    dtype=torch.long,    device=cfg.device).unsqueeze(0)
        done       = torch.tensor([int(done)],     dtype=torch.long,    device=cfg.device).unsqueeze(0)
        
        agent.ReplayBuffer.push(gid.clone(), state.clone(), action.clone(), reward.clone(), state_next.clone(), done.clone())
        agent.train()
        
        reward_episode += reward.item()
        
        state = state_next.clone()
        
        # step_total += 1
        
        if done:
            print(reward_episode)
            break

    print()


    with open(log_path, 'a') as log_file:
        log_file.write(f"{episode} {reward_episode:.9f} {epsilon:.9f}\n")
    

with open('./dqn.pkl' , 'wb') as f:
    pickle.dump(agent, f)















