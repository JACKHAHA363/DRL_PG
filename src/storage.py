import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RollOut(object):
    """
    storing one episode
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.noise = []

    def reset(self):
        """
        reset the storage
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.noise = []

    def add_transition(self, state, action, reward, noise):
        """
        Append transition
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.noise.append(noise)

