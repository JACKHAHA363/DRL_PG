import torch
import numpy as np

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

    def to_tensors(self):
        """
        Convert list to tensors
        :return
            t_states: [T, states_dim]. tensor
            t_actions: [T, action_dim]. tensor
            rewards: [T, 1]. Tensor
            noise: [T, 1]. Tensor
        """
        assert len(self.states) > 0 # we have something
        t_states = torch.from_numpy(np.array(self.states))
        t_actions = torch.from_numpy(np.array(self.actions))
        t_rewards = torch.from_numpy(np.array(self.rewards)).unsqueeze(-1)
        t_noise = torch.from_numpy(np.array(self.noise)).unsqueeze(-1)
        return t_states.float(), t_actions.float(), t_rewards.float(), t_noise.float()
