
import gym
from random import randrange
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import numpy as np
from typing import List, Tuple
import itertools
from iteration_utilities import duplicates , unique_everseen

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def new_grid(size):
    grid = np.zeros([size,size])
    return grid

def new_pos(existing, size):
    generated = (randrange(size), randrange(size))
    # if there are any duplicates, retry
    if np.any(np.in1d(generated, existing)):
        return new_pos(existing, size)
    return generated

def unique(a):
    return list(set(a))

class Action(Enum):
    North = 0
    South = 1
    East = 2
    West = 3

def move(pos, action):
    if action == Action.North or action == 0:
        return (pos[0], pos[1] - 1)
    elif action == Action.South or action == 1:
        return (pos[0], pos[1] + 1)
    elif action == Action.East or action == 2:
        return (pos[0] + 1, pos[1])
    elif action == Action.West or action == 3:
        return (pos[0] - 1, pos[1])
    else:
        raise Exception('not an action')

def limit_to_size(pos, grid_size):
    return tuple(map(lambda x: max(min(x, grid_size - 1), 0), pos))

class GridEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent = 2, grid_size = 8):
        self.num_agent = num_agent
        self.grid_size = grid_size
        self.reset()
        self.action_space = spaces.Discrete(4)
        self.state_space = num_agent * 4

    def step(self, actions):
        self.players = [limit_to_size(move(player, action), self.grid_size) for player, action in zip(self.players, actions)]

        states = self.get_state()
        is_at_goal = [player == goal for player, goal in zip(self.players, self.goals)]
        reward_is_at_goal = [-1 if x == False else 10 for x in is_at_goal]
        
        # detect a crash
        dup = list(unique_everseen(duplicates(self.players)))
        # if a player's position appears twice, add -20 to the current reward
        reward_is_crash = [-19 if (player in dup) else 0 for player in self.players]
        rewards = [a + b for a, b in zip(reward_is_at_goal, reward_is_crash)]

        done = True in is_at_goal 
        return (states, rewards, done)
 
    def reset(self):
        self.grid = new_grid(self.grid_size)
        self.players = unique([new_pos([], self.grid_size) for x in np.arange(self.num_agent)])
        self.goals = unique([new_pos(self.players, self.grid_size) for x in np.arange(self.num_agent)])
        # If we there are duplicate positions, retry
        if len(self.players) != self.num_agent or len(self.goals) != self.num_agent:
            return self.reset()
        return self.get_state()

    def get_state(self):
        zipped = list(map(flatten, list(zip(self.players, self.goals))))
        return list(map(lambda inner_array: list(map(lambda x: x / 8, inner_array)), zipped))

    def render(self, mode='human', close=False):
        annotated_grid = np.copy(self.grid)
        for index, player in enumerate(self.players):
            annotated_grid[player[1]][player[0]] = index + 1

        for index, goal in enumerate(self.goals):
            annotated_grid[goal[1]][goal[0]] = (index +1) * 10 + (index + 1)
        return annotated_grid

    
env = GridEnv(1)
print(env.render())
# print(env.players)
print(env.step([1,Action.East]))
# print(env.goals)
print("----")
print(env.players)
print(env.goals)