
import gym
from random import randrange
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import numpy as np
from typing import List, Tuple

grid_size = 8

def new_grid():
    grid = np.zeros([grid_size,grid_size])
    return grid

def new_pos():
    return (randrange(grid_size - 1), randrange(grid_size - 1))


class Action(Enum):
    North = 1
    South = 2
    East = 3
    West = 4

def move(pos, action):
    if action == Action.North:
        return (pos[0], pos[1] + 1)
    elif action == Action.South:
        return (pos[0], pos[1] - 1)
    elif action == Action.East:
        return (pos[0] - 1, pos[1])
    elif action == Action.West:
        return (pos[0] + 1, pos[1])
    else:
        raise Exception('not an action')

def limit_to_size(pos):
    return tuple(map(lambda x: min(x, grid_size), pos))

class GridEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    grid = new_grid()
    player_a = new_pos()
    player_b = new_pos()
    goal_a = new_pos()
    goal_b = new_pos()

    def __init__(self):
        pass
 
    def step(self, actions):
        self.player_a = limit_to_size(move(self.player_a, actions[0]))
        self.player_b = limit_to_size(move(self.player_b, actions[1]))

        states = [self.get_state(), self.get_state()]
        rewards = [
            10 if self.player_a == self.goal_a else -1,
            10 if self.player_b == self.goal_b else -1,
            ]
        done = self.player_a == self.goal_a or self.player_b == self.goal_b 
        return (states, rewards, done)
 
    def reset(self):
        self.grid = new_grid()
        self.player_a = new_pos()
        self.player_b = new_pos()
        self.goal_a = new_pos()
        self.goal_b = new_pos()
        return self.get_state()

    def get_state(self):
        current_state = np.copy(self.grid)
        current_state[self.player_a[0]][self.player_a[1]] = 1
        current_state[self.player_b[0]][self.player_b[1]] = 2

        current_state[self.goal_a[0]][self.goal_a[1]] = 11
        current_state[self.goal_b[0]][self.goal_b[1]] = 22
        return current_state
 
    def render(self, mode='human', close=False):
        return self.get_state()

    
env = GridEnv()
print(env.render())
print(env.step([Action.East,Action.West]))