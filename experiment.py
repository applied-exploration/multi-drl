 
class Experiment():
    def __init__(self, name, environment, agents, max_t=100, num_episodes = 1000):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.max_t = max_t
        self.num_episodes = num_episodes
