
from training import train
import logging
import uuid
import time

from utilities.monitor import save_scores, render_figure


class Experiment():
    def __init__(self, name, environment, agents, max_t=100, num_episodes=1000, goal = 0.):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.max_t = max_t
        self.num_episodes = num_episodes
        self.goal = goal

        self.id = uuid.uuid4()

        logging.basicConfig(filename='Logs/{}-{}.log'.format(time.strftime(
            "%Y-%m-%d_%H%M"),str(self.id)),
                    format='[%(levelname)s]: [%(asctime)s] [%(message)s]', datefmt='%m/%d/%Y %I:%M:%S %p')
        
        self.logger = logging.getLogger(str(self.id))

    def run(self):
        score_history, state_history = [], []
        try:
            print("Running experiment")
            score_history, state_history = train(env=self.environment,
                                                 agents=self.agents,
                                                 max_t=self.max_t,
                                                 num_episodes=self.num_episodes)
        except Exception as e:
            print("Encountered an error, going to log into file")
            self.save_error(e)
        finally:
            print("Ran experiments")
            return score_history, state_history


    def save(self, score_history=[], state_history=[], options=['scores', 'figures', 'states'], display = True, scores_window=0):
        if 'scores' in options: save_scores(score_history, config = self.agents[0].config)
        if 'states' in options: print(state_history)#save_states(score_history)
        render_figure(goal=self.goal, display=display, save= 'figures' in options, config = self.agents[0].config, scores_window=scores_window) 

    def save_error(self, error):
        self.logger.error(str(error))
