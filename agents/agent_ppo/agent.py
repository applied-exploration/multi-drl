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

from .memory import ReplayBuffer


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

    def __init__(self, state_size, action_size, config = PPOAgentConfig(), seed = 1, mode = '', max_t = 100):
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

        self.model = Model(state_size, action_size, seed = seed).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
    
        self.episode_rewards = []
        self.last_log_prob = None
        self.episode_log_probs = []

        self.mode = mode
        self.game_t = 0
        self.max_t = max_t
        # self.memory = Trajectories(seed, self.config.N_TRAJECTORIES, max_t = max_t)
        self.memory = ReplayBuffer(action_size = self.action_size, seed= seed, batch_size = self.config.N_TRAJECTORIES, buffer_size= self.config.N_TRAJECTORIES)
        self.run = 0
    
    def get_title(self):
        for_title = "PPO Agent"
        for_filename = "PPO"
        return for_title, for_filename

    def save(self, experiment_num, num_agent):
        torch.save(self.model.state_dict(), 'experiments/trained_agents/ppo_exp_{}__agent_{}_actor.pth'.format(experiment_num, num_agent))


    def act(self, state):
        p = self.model(state).detach().numpy()
        action = np.random.choice(np.arange(self.action_size), p = p)

        if self.mode is 'testing':  return action # this gives back probabilities
        elif self.mode is 'training': 
            # For the first steps collect create random steps to ensure we have a unique trajectory
            if self.game_t <= self.config.RANDOM_STEPS:
                action = np.random.choice(np.arange(self.action_size))
            self.game_t += 1
            return action

        else: sys.exit("Agent's Training/Testing mode not specified")
    


    def step(self, state, action, reward, next_state, done):
        prob = self.model(state).detach().numpy()

        self.memory.add(prob, state, action, reward, self.run) #, game_t = self.game_t, run = self.run)


    def reset(self):
        self.run += 1

        if self.run % self.config.N_TRAJECTORIES == 0:
            print("this is true")
            trajectories = self.memory.get()
            self.__learn(trajectories)
            self.memory.reset()
            self.game_t = 0 # reset the time step we are in the game
            self.run = 0

        


    def __learn(self, trajectories):
        old_probs, states, actions, rewards = trajectories

        for _ in range(self.config.SGD_epoch):
                
            L = -self.__clipped_surrogate(self.model, old_probs, states, actions, rewards, epsilon=self.config.EPSILON, beta=self.config.BETA)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

    def __clipped_surrogate(self, policy, old_probs, states, actions, rewards,
                      discount = 0.995, epsilon=0.1, beta=0.01):


        R_future = self.__discounted_future(rewards, discount)
        print(R_future.shape)
        R_norm_future = torch.tensor(self.__normalized_future(R_future),dtype=torch.int8, device=device)
        
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        rewards = torch.tensor(rewards, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.int8, device=device)
        
        # convert states to policy (or probability)
        new_probs = self.model(states) #pong_utils.states_to_prob(policy, states)
        #new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
        
        print("")
        print("")
        print("new_probs: ", new_probs.size())
        print("old_probs: ", old_probs.size())
        ratio = new_probs / (old_probs + 1e-7)
        print("ratio: ", ratio.size())
        g_clamped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        print("g_clamped size: ", g_clamped.size())
        g_PPO = torch.where(ratio < g_clamped, ratio, g_clamped)
        
        print("R_norm_future: ", R_norm_future.size())
        print("g_PPO: ",g_PPO.size())
        surrogates = (R_norm_future * g_PPO.T).mean()
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

