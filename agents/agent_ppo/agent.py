import numpy as np
import random
from collections import namedtuple, deque

import sys, os
sys.path.append(os.path.abspath('..'))
from agents.abstract_agent import Agent

from .model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim


class PPOAgentConfig:
    LR = 5e-4               # learning rate 
    GAMMA = 0.99
    SGD_epoch = 4
    BETA = 0.01
    EPSILON = 0.1
    RANDOM_STEPS = 4
    N_TRAJECTORIES = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config = REINFORCEAgentConfig(), seed = 1, mode = 'testing'):
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

        self.mode = mode
        self.game_t = 0
        self.memory = Trajectories()
    
    def get_title(self):
        for_title = "PPO Agent"
        for_filename = "PPO"
        return for_title, for_filename



    def act(self, state):
        if self.mode is 'testing': return self.model(state) # this gives back probabilities
        elif self.mode is 'training': 
            # For the first steps collect create random steps to ensure we have a unique trajectory
            action = np.random.choice(np.arange(self.action_size)) if self.game_t <= self.config.RANDOM_STEPS else self.model(state)
            self.game_t += 1
            return action

        else: sys.exit("Agent's Training/Testing mode not specified")
    


    def step(self, state, action, reward, next_state, done):
        prob = self.model(state)

        self.memory.add(prob, state, action, reward, game_t = self.game_t)


    def reset(self):
        self.game_t = 0 # reset the time step we are in the game

        if len(self.memory.memory) == self.config.N_TRAJECTORIES:
            trajectories = self.memory.get()
            self.__learn(trajectories)
            self.memory.reset()

        


    def __learn(self, trajectories):
        old_probs, states, actions, rewards = trajectories

        total_rewards = np.sum(rewards, axis=0)

        for _ in range(self.config.SGD_epoch):
                
            L = -self.__clipped_surrogate(self.model, old_probs, states, actions, rewards, epsilon=self.config.EPSILON, beta=self.config.BETA)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

    def __clipped_surrogate(self, policy, old_probs, states, actions, rewards,
                      discount = 0.995, epsilon=0.1, beta=0.01):


        R_future = self.__discounted_future(rewards, discount)
        R_norm_future = self.__normalized_future(R_future)
        
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        rewards = torch.tensor(rewards, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.int8, device=device)
        
        # convert states to policy (or probability)
        new_probs = self.model(states) #pong_utils.states_to_prob(policy, states)
        #new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
        
        ratio = new_probs / (old_probs + 1e-7)
        g_clamped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        g_PPO = torch.where(ratio < g_clamped, ratio, g_clamped)
        
        
        surrogates = (R_norm_future * g_PPO).mean()
        # include a regularization term
        # this steers new_policy towards 0.5
        # prevents policy to become exactly 0 or 1 helps exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        return torch.mean(surrogates + beta*entropy)

    def __discounted_future(self, rewards, discount):
        rewards= np.array(rewards)
        
        discounts = np.array([[discount**t for t in range(len(rewards))] for n in range(self.config.N_TRAJECTORIES)])

        trailing_rewards = rewards[::-1].cumsum(axis=0)[::-1]

        discounted_future_rewards = np.multiply(trailing_rewards.T, discounts)
        
        return discounted_future_rewards

    def __normalized_future(self, rewards):
        mean = np.mean(rewards, axis=1)
        std = np.std(rewards, axis=1) + 1.0e-10

        rewards_normalized = (rewards - mean[:,np.newaxis])/std[:,np.newaxis]
        
        return rewards_normalized


class Trajectories:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, seed, batch_size, N_TRAJECTORIES):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = N_TRAJECTORIES
        self.memory = [[]]#deque(maxlen=self.buffer_size)  # internal memory (deque)

        self.seed = random.seed(seed)
    
    def add(self, prob, state, action, reward, game_t ):
        """Add a new experience to memory."""

        self.memory[game_t]["probs"].append(prob)
        self.memory[game_t]["states"].append(state)
        self.memory[game_t]["actions"].append(action)
        self.memory[game_t]["rewards"].append(reward)

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def get(self):
        """Randomly sample a batch of experiences from memory."""
        probs = [memory_game_t.probs for memory_game_t in self.memory]
        states = [memory_game_t.probs for memory_game_t in self.memory]
        actions = [memory_game_t.probs for memory_game_t in self.memory]
        rewards = [memory_game_t.probs for memory_game_t in self.memory]

        return (probs, states, actions, rewards)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)