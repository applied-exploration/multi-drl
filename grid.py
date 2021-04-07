
import gym
from random import randrange
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum
import numpy as np
from typing import List, Tuple

GRID_SIZE = 8
NO_OF_PLAYERS = 2

def new_grid():
    grid = np.zeros([GRID_SIZE,GRID_SIZE])
    return grid

def new_pos():
    return (randrange(GRID_SIZE - 1), randrange(GRID_SIZE - 1))


class Action(Enum):
    North = 1
    South = 2
    East = 3
    West = 4

def move(pos, action):
    if action == Action.North:
        return (pos[0], pos[1] - 1)
    elif action == Action.South:
        return (pos[0], pos[1] + 1)
    elif action == Action.East:
        return (pos[0] + 1, pos[1])
    elif action == Action.West:
        return (pos[0] - 1, pos[1])
    else:
        raise Exception('not an action')

def limit_to_size(pos):
    return tuple(map(lambda x: max(min(x, GRID_SIZE), 0), pos))

class GridEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    grid = new_grid()
    players = [new_pos() for x in np.arange(NO_OF_PLAYERS)]
    goals = [new_pos() for x in np.arange(NO_OF_PLAYERS)]

    def __init__(self):
        pass
 
    def step(self, actions):
        self.players = [limit_to_size(move(player, action)) for player, action in zip(self.players, actions)]

        states = [self.get_state(), self.get_state()]
        is_at_goal = [player == goal for player, goal in zip(self.players, self.goals)]
        rewards = list(map(lambda x: -1 if x == False else 10, is_at_goal))
        done = True in is_at_goal 
        return (states, rewards, done)
 
    def reset(self):
        self.grid = new_grid()
        self.players = [new_pos() for x in np.arange(NO_OF_PLAYERS + 1)]
        self.goals = [new_pos() for x in np.arange(NO_OF_PLAYERS + 1)]
        return self.get_state()

    def get_state(self):
        annotated_grid = np.copy(self.grid)
        for index, player in enumerate(self.players):
            annotated_grid[player[1]][player[0]] = index + 1

        for index, goal in enumerate(self.goals):
            annotated_grid[goal[1]][goal[0]] = (index +1) * 10 + (index + 1)
        return annotated_grid
 
    def render(self, mode='human', close=False):
        return self.get_state()

    
env = GridEnv()
print(env.render())
print(env.players)
print(env.step([Action.East,Action.East])[0][1])
print(env.players)
