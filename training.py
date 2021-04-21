from agent_reinforce.agent import REINFORCEAgent
from agent_deepqn.agent import DeepQAgent
from environment.grid import GridEnv
from collections import deque
import numpy as np
from utilities.helper import flatten
from experiment import Experiment

def train(experiment: Experiment, scores_window=100, flatten_state=False, print_every = 20):
    env = experiment.environment
    agents = experiment.agents
    max_t = experiment.max_t
    num_episodes = experiment.num_episodes

    score_history = []
    scores_deque = deque(score_history[-scores_window:], maxlen=scores_window)
    last_running_mean = float('-inf')

    for episode in range(num_episodes):
        states = env.reset()
        scores = np.zeros(len(agents))

        for i in range(max_t):
            actions = []
            if flatten_state == True:
                actions = [agent.act(flatten(states)) for agent in agents]
            else:
                actions = [agent.act(state) for agent, state in zip(agents, states)]
            
            next_states, rewards, done = env.step(actions)
            if flatten_state == True:
                [agent.step(flatten(states), action, reward, flatten(next_states), done) for agent, action, reward in zip(agents, actions, rewards)]
            else:
                [agent.step(state, action, reward, next_state, done) for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, next_states)]

            scores += rewards

            states = next_states
            if done == True:
                break
        returns_in_episode = np.mean(scores)
        scores_deque.append(returns_in_episode)
        score_history.append(returns_in_episode)
        [agent.reset() for agent in agents]
        if episode > scores_window:
            if np.mean(scores_deque) > last_running_mean:
                    # print("")
                    # print('Last {} was better, going to save it'.format(scores_window))
                    [agent.save() for agent in agents]
                    last_running_mean = np.mean(scores_deque)
        if episode % print_every == 0:
            print("\r", 'Total score (averaged over agents) {} episode: {} | \tAvarage in last {} is {}'.format(episode, returns_in_episode, scores_window, np.mean(scores_deque)), end="")


    return score_history
