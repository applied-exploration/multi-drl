
import gym
from random import randrange
import random
from gym import error, spaces, utils
from gym.utils import seeding
from enum import IntEnum
import numpy as np
from typing import List, Tuple
import itertools
from iteration_utilities import duplicates , unique_everseen
from utilities.helper import unique, flatten


def new_grid(size):
    grid = np.zeros([size,size])
    return grid

def new_pos(existing, size):
    generated = (randrange(size), randrange(size))
    # if there are any duplicates, retry
    if np.any(np.in1d(generated, existing)):
        return new_pos(existing, size)
    return generated


class Action(IntEnum):
    North = 0
    South = 1
    East = 2
    West = 3

confusion_matrix = {
    Action.North: [Action.East, Action.West],
    Action.South: [Action.East, Action.West],
    Action.East: [Action.South, Action.North],
    Action.West: [Action.South, Action.North],
}

def move(pos, action, prob = 1):
    if prob != 1:
        action = np.random.choice([action] + confusion_matrix[action], 1, p=[prob, ((1 - prob)/2), ((1 - prob)/2)])

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

def limit_to_size(pos, grid_size):
    return tuple(map(lambda x: max(min(x, grid_size - 1), 0), pos))

class GridEnv(gym.Env):  
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent = 2, grid_size = 8, prob_right_direction = 1, fixed_start = False, fixed_goals = False, agents_fully_observable = False, render_board = False):
        self.num_agent = num_agent
        self.grid_size = grid_size
        self.prob_right_direction = prob_right_direction
        self.action_space = spaces.Discrete(4)
        if agents_fully_observable == True:
            self.state_space = num_agent * 4        
        else:
            self.state_space = 4
        self.fixed_start = fixed_start
        self.fixed_goals = fixed_goals
        self.render_board = render_board
        self.agents_fully_observable = agents_fully_observable
        self.players_starting = None
        self.goals_starting = None
        self.all_possibilities=list(itertools.product(range(0,self.grid_size), repeat = 2))
        self.working_possibilites = self.all_possibilities.copy()

        self.init_goals()
        self.init_agents()

        self.reset()

    def init_goals(self):
        self.goals = []
        for _ in range(0, self.num_agent):
            goal = random.choice(self.working_possibilites)
            self.working_possibilites.remove(goal)
            self.goals.append(goal)
        self.goals_starting = self.goals.copy()

    def init_agents(self):
        self.players = []
        for _ in range(0, self.num_agent):
            start = random.choice(self.working_possibilites)
            self.working_possibilites.remove(start)
            self.players.append(start)
        self.players_starting = self.players.copy()

    def step(self, actions):
        self.players = [limit_to_size(move(player, action, self.prob_right_direction), self.grid_size) for player, action in zip(self.players, actions)]

        states = self.__get_state()
        is_at_goal = [player == goal for player, goal in zip(self.players, self.goals)]
        reward_is_at_goal = [-1 if x == False else 20 for x in is_at_goal]
        
        # detect a crash
        dup = list(unique_everseen(duplicates(self.players)))
        # if a player's position appears twice, add -20 to the current reward
        reward_is_crash = [-19 if (player in dup) else 0 for player in self.players]
        rewards = [a + b for a, b in zip(reward_is_at_goal, reward_is_crash)]

        done = True in is_at_goal 

        if self.render_board:
            print("{}".format(self.render()))
        return (states, rewards, done)
 
    def reset(self):
        self.grid = new_grid(self.grid_size)

        if self.fixed_goals == False or self.goals_starting == None:
            self.init_goals()
        elif self.fixed_goals == True and self.goals_starting != None:
            self.goals = self.goals_starting.copy()

        if self.fixed_start == False or self.players_starting == None:
            self.init_agents()
        elif self.fixed_start == True and self.players_starting != None:
            self.players = self.players_starting.copy()
        
        self.working_possibilites = self.all_possibilities.copy()
        return self.__get_state()

    def __get_state(self):
        players_goals = list(map(flatten, list(zip(self.players, self.goals))))
        players_goals = list(map(lambda inner_array: list(map(lambda x: x / self.grid_size, inner_array)), players_goals))
        if self.agents_fully_observable == False:
            return players_goals
        else:
            return [flatten(players_goals) for i in range(self.num_agent)]


    def render(self, mode='human', close=False):
        annotated_grid = np.copy(self.grid)
        for index, player in enumerate(self.players):
            annotated_grid[player[1]][player[0]] = index + 1

        for index, goal in enumerate(self.goals):
            annotated_grid[goal[1]][goal[0]] = (index +1) * 10 + (index + 1)
        return annotated_grid

    