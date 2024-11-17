# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:10:46 2024

@author: jyzhang
"""

import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

log_file = './log.txt'
log = np.loadtxt(log_file)

episodes = log[:, 0]
rewards = log[:, 1]

# Smooth the rewards using a moving average
window_size = 100  
smoothed_rewards = moving_average(rewards, window_size)
smoothed_episodes = episodes[window_size-1:]  # Adjust episodes to match the length of smoothed_rewards

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, label='Episode Reward', alpha=0.5)
plt.plot(smoothed_episodes, smoothed_rewards, label='Smoothed Reward', color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode vs Reward')
plt.legend()
plt.grid(True)

plt.savefig('./reward.png')
plt.close()





