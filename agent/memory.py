
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from constants import * 
 
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=BUFFER_SIZE)  # internal memory (deque)
        self.batch_size = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "state_full", "action", "reward", "next_state", "next_state_full", "done"])

        self.seed = random.seed(seed)
    
    def add(self, state, state_full, action, reward, next_state, next_state_full, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_full, action, reward, next_state, next_state_full, done)
        # print("Adding: ", e)
        self.memory.append(e)
    
    def reset(self):
        self.memory = deque(maxlen=BUFFER_SIZE)  # internal memory (deque)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # print("Sampled: ", experiences)
        print("Experience")
        print(experiences[0].state_full)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        states_full = torch.from_numpy(np.stack([e.state_full for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        next_states_full = torch.from_numpy(np.vstack([e.next_state_full for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        
        print(states_full.shape)
        return (states, states_full, actions, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)