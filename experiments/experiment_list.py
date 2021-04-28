import numpy as np
import itertools

""" 
--- Experiment Batch 1 ---

Question: Will a DQN Agent and a REINFORCE Agent converage on a 
    * fixed goal & fixed player
    * random goal & fixed player
    * fixed goal & random player
    * random goal & random player
Hypothesis: Both Agents' hyperparameters can be tuned to work in all 3 scenarios.
Setup: 
    Grid-size: 5 x 5
    Num Agents: 1
    Stochastic: 1.0

Number of experiments: 1 x 2 x 2 x 3 x 2 = 24

""" 

# --- environment params --- #
__num_agent = np.array([1])
__agents_start = np.array([True, False])
__goals_start = np.array([True, False])
__prob_right_direction = np.array([1])
__grid_size = np.array([5])

# --- agent params --- #
__network_dqn = [[16], [32], [64]]
__network_rei = [[16], [32], [64]]

__exp1_dqn = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_dqn]
__exp1_rei = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_rei]

exp1_dqn = list(itertools.product(*__exp1_dqn))
exp1_rei = list(itertools.product(*__exp1_rei))



""" 
--- Experiment Batch 2 ---

Question: Will a DQN Agent and a REINFORCE Agent converage on a stochastic environment
Hypothesis: Both Agents' hyperparameters can be tuned to work in a stochastic environment.
Setup: 
    Grid-size: 5 x 5
    Num Agents: 1
    Agents, Goals randomized
    Stochastic: 0.7

Number of experiments: 3 + 3 = 6
""" 

# --- environment params --- #
__num_agent = np.array([1])
__agents_start = np.array([False])
__goals_start = np.array([False])
__prob_right_direction = np.array([0.7])
__grid_size = np.array([5])

# --- agent params --- #
__network_dqn = [[16], [32], [64]]
__network_rei = [[16], [32], [64]]

__exp2_dqn = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_dqn]
__exp2_rei = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_rei]

exp2_dqn = list(itertools.product(*__exp2_dqn))
exp2_rei = list(itertools.product(*__exp2_rei))



""" 
--- Experiment Batch 3 ---

Question: Will a DQN Agent and a REINFORCE Agent converage on a multi-agent environment?
Hypothesis: Both Agents' hyperparameters can be tuned to work in a multi-agent environment.
Setup: 
    Grid-size: 5 x 5
    Num Agents: 1
    Agents, Goals randomized
    Stochastic: 0.7

Number of experiments: 3 + 3 = 6
""" 

# --- environment params --- #
__num_agent = np.array([2, 3])
__agents_start = np.array([False])
__goals_start = np.array([False])
__prob_right_direction = np.array([0.7])
__grid_size = np.array([5])

# --- agent params --- #
__network_dqn = [[16, 16], [32, 32], [64, 64]]
__network_rei = [[16], [32], [64]]

__exp3_dqn = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_dqn]
__exp3_rei = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_rei]

exp3_dqn = list(itertools.product(*__exp3_dqn))
exp3_rei = list(itertools.product(*__exp3_rei))



""" 
--- Experiment Batch 4 ---

Question: Will a pretrained DQN and REINFORCE Agent converge on a larger map?
Hypothesis: Both Agents can perform well on a larger environment.
Setup: 
    Grid-size: [8x8, 12x12]
    Num Agents: 1
    Agents, Goals randomized
    Stochastic: 0.7

Number of experiments: 4 
""" 

# --- environment params --- #
__num_agent = np.array([2, 3])
__agents_start = np.array([False])
__goals_start = np.array([False])
__prob_right_direction = np.array([0.7])
__grid_size = np.array([8, 12])

__network_dqn = [[16]] ## Should be selected from previous best!
__network_rei = [[16]] ## Should be selected from previous best!


__exp4_dqn = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_dqn]
__exp4_rei = [__num_agent, __grid_size, __agents_start, __goals_start, __prob_right_direction, __network_rei]

exp4_dqn = list(itertools.product(*__exp4_dqn))
exp4_rei = list(itertools.product(*__exp4_rei))



