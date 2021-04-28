import numpy as np
import random

from utilities.helper import flatten



import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")     # Training on GPU or CPU


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, seed, batch_size, buffer_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["probs", "states", "actions", "rewards"])

        self.seed = random.seed(seed)
    
    def add(self, prob, state, action, reward, run):
        """Add a new experience to memory."""
        e = self.experience(prob, state, action, reward)
        if run < len(self.memory): self.memory[run].append(e)
        else: self.memory.append([e])
    
    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def get(self):
        """Randomly sample a batch of experiences from memory."""
        #trajectories = random.sample(self.memory, k=self.batch_size)
        experiences = self.memory[0]
        
        probs = torch.from_numpy(np.vstack([e.probs for e in experiences if e is not None])).float().to(DEVICE)
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(DEVICE)
        print("rewards: ", rewards.size())

        return (probs, states, actions, rewards)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# class Trajectories:
#     """Fixed-size buffer to store experience tuples."""

#     def __init__(self, seed, N_TRAJECTORIES, max_t):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """

#         self.buffer_size = N_TRAJECTORIES
#         self.max_t = max_t
#         # self.memory = np.zeros((max_t, 4))
#         self.memory = np.ndarray((N_TRAJECTORIES, max_t, 4))
#         self.seed = random.seed(seed)
    
#     def add(self, prob, state, action, reward, game_t, run):
#         """Add a new experience to memory."""
#         if run < len(self.memory): 
#             if game_t < len(self.memory[run]): 
#                 self.memory[run][game_t][0].append(prob)
#                 self.memory[run][game_t][1].append(state)
#                 self.memory[run][game_t][2].append(action)
#                 self.memory[run][game_t][3].append(reward)
#             else:
#                 self.memory[run].append([prob,state,action,reward])
#         else:
#             self.memory.append([[prob,state,action,reward]])

#         print(self.memory[:1][0])
#         print(self.memory[:1][1])
#         print(self.memory[:1][2])


#     def reset(self):
#         self.memory = []

#     def get(self):
#         """Randomly sample a batch of experiences from memory."""
        
#         probs = flatten([memory_game_t[0] for memory_game_t in self.memory])
#         states = flatten([memory_game_t[1] for memory_game_t in self.memory])
#         actions = flatten([memory_game_t[2] for memory_game_t in self.memory])
#         rewards = flatten([memory_game_t[3] for memory_game_t in self.memory])

#         return (probs, states, actions, rewards)

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)