import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from .helper import get_constant_string
# from constants import *             # Capital lettered variables are constants from the constants.py file

import os


def calculate_moving_avarage(scores, num_agent=1, scores_window=100):
    single_agent_returns = scores
    moving_avarages = [np.convolve(scores[i], np.ones(
        scores_window)/scores_window, mode='valid') for i in range(num_agent)]

    return moving_avarages


def render_figure(scores, config, scores_window=0, num_agent=1, path="", goal=0, save=False, display= True):
    if len(path) < 1:
        path = config.SAVE_EXP_PATH

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- Plot labels --- #
    hyperparameter_string, for_filename = get_constant_string(config)
    plt.title(hyperparameter_string)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    # --- Plot lines --- #
    plt.plot(np.arange(1, len(scores)+1), scores)

    if config.scores_window > 0:
        moving_avarages = calculate_moving_avarage(
            [scores], num_agent, scores_window=scores_window)

        for i_agent in range(len(moving_avarages)):
            plt.plot(np.arange(len(moving_avarages[i_agent]) + scores_window)[scores_window:], moving_avarages[i_agent], 'g-')

    if goal > 0.:
        plt.axhline(y=goal, color='c', linestyle='--')

    # --- Save and Display --- #
    if save:
        plt.savefig("{}Figure_{}_{}.jpg".format(path, time.strftime(
            "%Y-%m-%d_%H%M"), for_filename), bbox_inches='tight')

    if display: plt.show()




def save_scores(scores, config, path=""):
    if len(path) < 1:
        path = 'experiments/'

    if not os.path.exists(path):
        print("Directory doesn't exist, going to create one first")
        os.makedirs(path)

    _, for_filename = get_constant_string()

    with open("{}Scores_{}_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M"), for_filename), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(scores)

    print("Scores saved!")


def read_scores(config, network_name=''.format(time.strftime("%Y-%m-%d_%H%M")), path=''):
    if len(path) < 1:
        path = config.SAVE_EXP_PATH

    if os.path.exists(path):

        # _, for_filename  = get_constant_string()

        with open("{}{}.csv".format(path, network_name), newline='') as f:
            reader = csv.reader(f)
            read_score_history = list(reader)[0]

        parsed = [float(i) for i in read_score_history]

        return parsed
