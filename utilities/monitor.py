import numpy as np
import matplotlib.pyplot as plt
import csv
import time
#from .helper import get_constant_string
# from constants import *             # Capital lettered variables are constants from the constants.py file

import os


def calculate_moving_avarage(scores, num_agent=1, scores_window=100):
    if num_agent < 2: single_agent_returns = np.array(scores)
    else: single_agent_returns = np.transpose(np.array(scores))
    moving_avarages = [np.convolve(single_agent_returns[i], np.ones(scores_window)/scores_window, mode='valid') for i in range(num_agent)]

    return moving_avarages


def calculate_max(scores):
    best_score = []

    for i, episode_score in enumerate(scores):
        best_score.append(np.max(episode_score))

    return best_score


def render_figure(scores, agents, env_params, name="", scores_window=0, path="", goal=0, save=False, display= True):
    if len(path) < 1:
        path = 'experiments/saved/'

    # fig, (ax, tb) = plt.subplots(nrows=1, ncols=2)
    fig = plt.figure()

    ax = fig.add_subplot(1, 3, (1, 2))
    tb1 = fig.add_subplot(3, 3, 3)
    tb2 = fig.add_subplot(3, 3, 6)
    tb3 = fig.add_subplot(3, 3, 9)

    # --- Plot labels --- #
    for_title, for_filename, for_table, for_id = agents[0].get_title()


    ax.set_title(for_title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')

    fig.text(0.975, 0.1, for_id, size=7, color='gray', 
        horizontalalignment='right',
        verticalalignment='top')




    # --- Plot scores --- #
    if len(agents)>1: # multiple agents
        accumulated_by_agent = np.transpose(np.array(scores))
        for i_agent in range(len(agents)):
            ax.plot(np.arange(1, len(accumulated_by_agent[i_agent])+1), accumulated_by_agent[i_agent])
    else: ax.plot(np.arange(1, len(scores)+1), scores)

    # --- Plot moving avarages --- #
    best_avg_score = None
    episode_achieved = 0
    final_avarage = 0
    highest = 0

    if scores_window > 0:
        moving_avarages = []
        if len(agents)>1: 
            moving_avarages = calculate_moving_avarage(scores, len(agents), scores_window=scores_window)

            best_of_two = calculate_moving_avarage([calculate_max(scores)], 1, scores_window=scores_window)

            episode_achieved = np.argmax(best_of_two[0])
            best_avg_score = best_of_two[0][episode_achieved]
            episode_achieved += scores_window
            final_avarage = best_of_two[0][-1]
            highest = max(scores[0])

            ax.plot(np.arange(len(best_of_two[0]) + scores_window)[scores_window:], best_of_two[0], 'k-')
        else:             
            moving_avarages = calculate_moving_avarage(scores, len(agents), scores_window=scores_window)
            
            episode_achieved = np.argmax(moving_avarages[0])
            best_avg_score = moving_avarages[0][episode_achieved]
            episode_achieved += scores_window
            final_avarage = moving_avarages[0][-1]
            highest = max(scores)
        
        for i_agent in range(len(moving_avarages)):
            ax.plot(np.arange(len(moving_avarages[i_agent]) + scores_window)[scores_window:], moving_avarages[i_agent], 'g-')
        
    if goal > 0.: ax.axhline(y=goal, color='c', linestyle='--')

        # --- Plot table --- #
    # for env #
    tb1.axis('tight')
    tb1.axis("off")
    rows = env_params[0]
    columns = ['Env']
    cell_text = env_params[1]
    tb1.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns, 
                      loc='center right')

    # for agent #
    tb2.axis('tight')
    tb2.axis("off")
    rows = for_table[0]
    columns = ['Agent']
    cell_text = for_table[1]
    tb2.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns, 
                      loc='center right')


    # for scores #
    tb3.axis('tight')
    tb3.axis("off")
    rows = ["best in {}".format(scores_window), "ep. achieved", "final avg.", "highest ep."]
    columns = ['Scores']
    cell_text = [['{:.1f}'.format(best_avg_score)], [episode_achieved], ['{:.1f}'.format(final_avarage)], [highest]]
    tb3.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns, 
                      loc='center right')



    fig.tight_layout()


    # --- Save and Display --- #
    if save: fig.savefig("{}{}_figure_{}.jpg".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), bbox_inches='tight')
    if display: fig.show()




def save_scores(scores, agents, name="",  path=""):
    if len(path) < 1:
        path = 'experiments/saved/'

    if not os.path.exists(path):
        print("Directory doesn't exist, going to create one first")
        os.makedirs(path)

    for_title, for_filename, for_table, for_id = agents[0].get_title()

    with open("{}{}_scores_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(scores)

    print("Scores saved!")


def read_scores(network_name=''.format(time.strftime("%Y-%m-%d_%H%M")), path=''):
    if len(path) < 1:
        path = 'experiments/saved/'

    if os.path.exists(path):

        # _, for_filename  = get_constant_string()

        with open("{}{}.csv".format(path, network_name), newline='') as f:
            reader = csv.reader(f)
            read_score_history = list(reader)[0]

        parsed = [float(i) for i in read_score_history]

        return parsed

def save_states(states, name="", path=""):
    if len(path) < 1:
        path = 'experiments/saved/'

    with open("{}{}_states_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(states)