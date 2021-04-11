
import gym
from math import sqrt, cos, sin
import random
from gym import error, spaces, utils
from gym.utils import seeding
from enum import IntEnum
import numpy as np
from typing import List, Tuple
import itertools
from iteration_utilities import duplicates , unique_everseen
from gym import spaces
import matplotlib.pyplot as plt


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def new_grid(size):
    grid = np.zeros([size,size])
    return grid

def new_pos(existing, size):
    generated = (random.uniform(0.0, float(size)), random.uniform(0.0, float(size)))
    # if there are any duplicates, retry
    if np.any(np.in1d(generated, existing)):
        return new_pos(existing, size)
    return generated

def unique(a):
    return list(set(a))

def v_length(vec):
    return sqrt(vec[0]**2 + vec[1]**2)

def v_subtract(lhs, rhs):
    return (lhs[0] - lhs[1], rhs[0] - rhs[1])

def v_add(lhs, rhs):
    return (lhs[0] + lhs[1], rhs[0] + rhs[1])

def v_within_range(vec1, vec2, radius):
    return v_length(v_subtract(vec1, vec2)) < radius

def v_list_within_range(vecs, radius):
    pairs_within_range = [combination if v_within_range(combination[0], combination[1], radius) else None for combination in itertools.combinations(vecs, 2)]
    return list(filter(None, pairs_within_range))

def rotate_origin_only(vec, radians):
    x, y = vec
    xx = x * cos(radians) + y * sin(radians)
    yy = -x * sin(radians) + y * cos(radians)

    return (xx, yy)

def move(pos, action, prob = 1):
    print(action)
    print(pos)
    if prob != 1:
        # add random rotation up to 1 radian and the original action vector, and sample with probabiliby `prob`
        possible_actions = [action, rotate_origin_only(action, random.uniform(0.0, 1.0))]
        rand_choice = np.random.choice(len(possible_actions), 1, p=[prob, 1 - prob])[0]
        action = possible_actions[rand_choice]

    return v_add(pos, action)

def limit_to_size(pos, grid_size):
    return tuple(map(lambda x: max(min(x, grid_size - 1), 0), pos))

class TwoDSurfaceEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent = 2, grid_size = 8, prob_right_direction = 1):
        self.num_agent = num_agent
        self.grid_size = grid_size
        self.prob_right_direction = prob_right_direction
        self.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, num_agent), dtype=np.float32)
        self.state_space = spaces.Box(low=0.0, high=grid_size, shape=(2, num_agent), dtype=np.float32)

    def step(self, actions):
        self.players = [limit_to_size(move(player, action, self.prob_right_direction), self.grid_size) for player, action in zip(self.players, actions)]

        states = self.get_state()
        # this has to be an area
        is_at_goal = [v_within_range(player, goal, 0.05) for player, goal in zip(self.players, self.goals)]
        reward_is_at_goal = [-1 if x == False else 10 for x in is_at_goal]
        
        # detect a crash - if the agent is within radius
        players_within_range = flatten(v_list_within_range(self.players, 0.05))
        # if a player's position appears twice, add -20 to the current reward
        reward_is_crash = [-19 if (player in players_within_range) else 0 for player in self.players]
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
        print(self.players)
        print(self.goals)

        plt.scatter(list(zip(*self.players))[0], list(zip(*self.players))[1], c = 'blue')
        plt.scatter(list(zip(*self.goals))[0], list(zip(*self.goals))[1], c = 'red')
        plt.title("2D Surface")

        axis = plt.gca()
        axis.set_xlim(0, self.grid_size)
        axis.set_ylim(0, self.grid_size)
        plt.show()

    
# env = TwoDSurfaceEnv(2, 8, 0.1)
# # print(env.players)
# # print(env.goals)

# # env.step([(1., 2.), (2.,2.)])
# env.render()
