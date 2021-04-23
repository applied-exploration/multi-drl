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

# num_agent = np.array([2])
# agents_start=np.array([True])
# goals_start = np.array([False, True])
# prob_right_direction = np.array([1])
# grid_size = np.array([3])


num_agent = np.array([1, 2, 3, 4])
agents_start = np.array([True])
goals_start = np.array([True, False])
prob_right_direction = np.array([1, 0.7])
grid_size = np.array([4, 8, 12])
#fully_observable = np.array([True, False])

possible_starts = [(1,1),(1,3),(2,3),(3,3),(4,4),(6,7),(5,7),(3,7),(5,9),(5,10),(2,10)]
possible_goals = [(3,1),(3,2),(2,1),(0,0),(3,6),(6,3),(1,7),(0,5),(0,9),(10,5),(10,2)]

# --- AGENTS --- #
# DDPG
actor_critic = [[16], [32], [16, 16], [32, 32], [64, 64]]

# PPO/REINFORCE
network = [[16, 16], [32, 32], [64, 64]]

# DQN
network_dqn = [[16], [32], [64]]


config_ddpg = [num_agent, grid_size, agents_start,
               goals_start, prob_right_direction, actor_critic]
config_ppo = [num_agent, grid_size, agents_start,
              goals_start, prob_right_direction, network]
config_dqn = [num_agent, grid_size, agents_start,
              goals_start, prob_right_direction, network_dqn]

exp_config_ddpg = list(itertools.product(*config_ddpg))
exp_config_ppo = list(itertools.product(*config_ppo))
exp_config_dqn = list(itertools.product(*config_dqn))

