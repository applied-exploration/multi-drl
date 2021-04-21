""" 
ENVIRONMENTS
======

environments = Grid
num_players = 1, 2, 3, 4
fixed, semi-fixed, random 
4x4, 8x8, 12x12
stochastic, not stochastic

1 x 4 x 3 x 3 x 2 = 72

"""

"""
AGENTS
======

PPO
[16, 16], [32, 32], [64, 64]

DQN
[16, 16], [32, 32], [64, 64]

DDPG 
actor  = [16, 16], [32, 32], [64, 64]
critic = [16, 16], [32, 32], [64, 64]

learning_rate = 0.0001, 0.001, 0.01
learning_rate = 0.001, 0.001, 0.01

3 x 3 x 3 = 27
"""


# --- ENVIRONMENT --- #
import numpy as np
import itertools

num_players = np.array([1, 2, 3, 4])
agents_start = np.array([None, (1, 1), (1, 1)], dtype=object)
goals_start = np.array([None, None, (3, 3)], dtype=object)
prob_right_direction = np.array([1, 0.7])

# --- AGENTS --- #
# DDPG
actor_critic = [(16,16), (32,32), (64,64)]
# PPO
network = [(16,16), (32,32), (64,64)]
# DQN
network_dqn = [(16,16), (32,32), (64,64)]


config_ddpg = [num_players, agents_start, goals_start, prob_right_direction, actor_critic]
config_ppo = [num_players, agents_start, goals_start, prob_right_direction, network]
config_dqn = [num_players, agents_start, goals_start, prob_right_direction, network_dqn]

print(len(list(itertools.product(*configs))))
