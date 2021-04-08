
import gym
from random import randrange
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import numpy as np
from typing import List, Tuple
import itertools

GRID_SIZE = 8
NO_OF_PLAYERS = 2

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def new_grid():
    grid = np.zeros([GRID_SIZE,GRID_SIZE])
    return grid

def new_pos(existing):
    generated = (randrange(GRID_SIZE), randrange(GRID_SIZE))
    # if there are any duplicates, retry
    if np.any(np.in1d(generated, existing)):
        return new_pos(existing)
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

def limit_to_size(pos):
    return tuple(map(lambda x: max(min(x, GRID_SIZE - 1), 0), pos))

class GridEnv(gym.Env):  
    metadata = {'render.modes': ['human']}


    def __init__(self):
        self.reset()
        self.num_agent = NO_OF_PLAYERS
        self.action_space = spaces.Discrete(4)
        self.state_space = GRID_SIZE * GRID_SIZE
 
    def step(self, actions):
        self.players = [limit_to_size(move(player, action)) for player, action in zip(self.players, actions)]

        states = [self.get_state(), self.get_state()]
        is_at_goal = [player == goal for player, goal in zip(self.players, self.goals)]
        rewards = list(map(lambda x: -1 if x == False else 10, is_at_goal))
        done = True in is_at_goal 
        return (states, rewards, done)
 
    def reset(self):
        self.grid = new_grid()
        self.players = unique([new_pos([]) for x in np.arange(NO_OF_PLAYERS)])
        self.goals = unique([new_pos(self.players) for x in np.arange(NO_OF_PLAYERS)])
        # If we there are duplicate positions, retry
        if len(self.players) != 2 or len(self.goals) != 2:
            return self.reset()
        return self.get_state()

    def get_state(self):
        return list(map(lambda x: x / 8, flatten([flatten(self.players), flatten(self.goals)])))
 
    def render(self, mode='human', close=False):
        annotated_grid = np.copy(self.grid)
        for index, player in enumerate(self.players):
            annotated_grid[player[1]][player[0]] = index + 1

        for index, goal in enumerate(self.goals):
            annotated_grid[goal[1]][goal[0]] = (index +1) * 10 + (index + 1)
        return annotated_grid

    
env = GridEnv()
print(env.render())
print(env.players)
print(env.step([1,Action.East])[0][1])
print(env.goals)

