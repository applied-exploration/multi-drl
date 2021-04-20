## Deep Deterministic Policy Gradients ##
from model import Actor, Critic    # These are our models
import numpy as np
import random                       # Used for random seed
import copy                         # This is used for the mixing of target and local model parameters

from constants import *             # Capital lettered variables are constants from the constants.py file


import torch
import torch.nn.functional as F
import torch.optim as optim

class DDPG_Agent:
    def __init__(self, state_size, action_size, random_seed, config = DDPGAgentConfig(), actor_hidden= [400, 300], critic_hidden = [400, 300], id=0, num_agent=1):
        super(DDPG_Agent, self).__init__()
        self.config = config

        self.actor_local = Actor(state_size, action_size, random_seed, hidden_layer_param=actor_hidden).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_layer_param=actor_hidden).to(DEVICE)
        self.critic_local = Critic(state_size, action_size, random_seed, hidden_layer_param=critic_hidden).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed, hidden_layer_param=critic_hidden).to(DEVICE)

        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=self.config.LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=self.config.LR_CRITIC)



        self.seed = random.seed(random_seed)
        self.id=id

        print("")
        print("--- Agent {} Params ---".format(self.id))
        print("Going to train on {}".format(DEVICE))
        print("Learning Rate:: Actor: {} | Critic: {}".format(self.config.LR_ACTOR, self.config.LR_CRITIC))
        print("Replay Buffer:: Buffer Size: {} | Sampled Batch size: {}".format(self.config.BUFFER_SIZE, self.config.BATCH_SIZE))
    
    # def act(self, state):
    #     state = torch.from_numpy(state).float().to(DEVICE)

    #     self.actor_local.eval()
    #     with torch.no_grad():
    #         actions = self.actor_local(state).cpu().data.numpy()
    #     self.actor_local.train()

    #     return actions
    
    
    def act(self, obs, noise=0.0):
        obs = obs.to(DEVICE)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(obs) #+ noise*self.noise.noise()

        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(DEVICE)

        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(obs) #+ noise*self.noise.noise()

        return action


    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences
 

        # ---                   Teach Critic (with TD)              --- #
        recommended_actions = self.actor_target(next_states)
        Q_nexts = self.critic_target(next_states, recommended_actions)
        Q_targets = (rewards + self.config.GAMMA * Q_nexts * (1 - dones))                 # This is what we actually got from experience
        Q_expected = self.critic_local(states, actions)                       # This is what we thought the expected return of that state-action is.
        critic_loss = CRITERION(Q_targets, Q_expected)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()


        # ---                   Teach Actor                          --- #
        next_actions = self.actor_local(states)
        # Here we get the value of each state-actions. 
        # This will be backpropagated to the weights that produced the action in the actor network. 
        # Large values will make weights stronger, smaller values (less expected return for that state-action) weaker
        actor_loss = -self.critic_local(states, next_actions).mean()            
        

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        # Mix model parameters in both Actor and Critic #
        self.soft_update(self.actor_local, self.actor_target) 
        self.soft_update(self.critic_local, self.critic_target) 
    
    def soft_update(self, local, target):
        """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.config.TAU*local_param.data + (1.0-self.config.TAU)*target_param.data)

