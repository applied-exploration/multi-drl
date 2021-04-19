from model import Model
import numpy as np
import random
from collections import namedtuple, deque

import sys, os
sys.path.append(os.path.abspath('..'))
from abstract_agent import Agent

from model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 5e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class REINFORCEAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed = 1):
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

        self.model = Model(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
    


    def act(self, state):
        return self.model.act(state)

    def learn(self, rewards, gamma, saved_log_probs):
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def reset(self):
        pass