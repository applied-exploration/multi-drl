# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_w_memory import DDPG_Agent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
import numpy as np
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import *  


class MADDPG:
    def __init__(self, state_size, action_size, random_seed=32, num_agent = 2):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPG_Agent(state_size, action_size, random_seed, actor_hidden=[128, 64], critic_hidden=[128, 64], id=i) for i in range(num_agent)]

        
        #state_size, action_size, random_seed, actor_hidden= [400, 300], critic_hidden = [400, 300]
        #def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2)


        self.discount_factor = GAMMA
        self.tau = TAU
        self.iter = 0

    def reset(self, state_size, action_size, random_seed=32, num_agent = 2):
        self.maddpg_agent = [DDPG_Agent(state_size, action_size, random_seed, actor_hidden=[128, 64], critic_hidden=[128, 64], id=i) for i in range(num_agent)]

    def step(self, state, action, reward, next_state, done):
        flattened_state = [item for sublist in state for item in sublist]
        flattened_next_state = [item for sublist in next_state for item in sublist]
        for i, agent in enumerate(self.maddpg_agent):
            agent.step(flattened_state, action[i], reward[i], flattened_next_state, done)

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        flattened_state = torch.Tensor([item for sublist in obs_all_agents for item in sublist])
        actions = [agent.act(flattened_state, noise) for agent in self.maddpg_agent]
        return actions




