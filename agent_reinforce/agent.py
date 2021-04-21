import numpy as np
import random
from collections import namedtuple, deque

import sys, os
sys.path.append(os.path.abspath('..'))
from abstract_agent import Agent

from .model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim


class REINFORCEAgentConfig:
    LR = 5e-4               # learning rate 
    GAMMA = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class REINFORCEAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config = REINFORCEAgentConfig(), seed = 1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = config

        self.model = Model(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
    
        self.episode_rewards = []
        self.last_log_prob = None
        self.episode_log_probs = []
        

    def act(self, state):
        action, log_prob = self.model.act(state)
        self.last_log_prob = log_prob
        return action

    def step(self, state, action, reward, next_state, done):
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(self.last_log_prob)


    def reset(self):
        if len(self.episode_rewards) == 0: return
        self.__learn(self.episode_rewards, self.episode_log_probs)
        self.episode_log_probs = []
        self.episode_rewards = []

    def __learn(self, rewards, episode_log_probs):
        discounts = [self.config.GAMMA**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in episode_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()