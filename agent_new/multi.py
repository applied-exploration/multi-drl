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

    def step(self, state, action, reward, next_state, done):
        for i, agent in enumerate(self.maddpg_agent):
            agent.step(state[i], action[i], reward[i], next_state[i], done)

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions




