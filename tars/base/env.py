import gym
from tars.base.configurable import Configurable


class Env(gym.Env, Configurable):
    def __init__(self, observation_space, action_space):
        gym.Env.__init__(self)
        Configurable.__init__(self)
        self.observation_space = observation_space
        self.action_space = action_space

    def seed(self, seed=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
