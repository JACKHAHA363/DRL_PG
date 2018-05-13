import torch
import numpy as np
from .utils import compute_returns
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


class Dataset(object):
    """
    Used to sample batch of transition and store data into tensor format.
    Can be used to iterate. Everything is not used for backprop
    """
    def __init__(self, memory, actor, critic, gamma):
        """
        All input are tensors. Compute returns, values, and evaluate action as well.
        :param t_states: memory. Rollout object.
        :param actor: policy network
        :param critic: value network
        :param gamma: discounting rate
        """
        self.total_size = len(memory.states)

        # convert to tensors
        self.states, self.actions, self.rewards, self.noise = memory.to_tensors()

        # compute returns [T, 1]
        self.returns = compute_returns(memory, gamma)
        self.returns = torch.from_numpy(self.returns).float().unsqueeze(-1)

        # forward and get state value
        self.values = critic(self.states).detach()

        # forward and get action probs [T, 1]
        self.means, self.logvars = actor(self.states)
        self.means = self.means.detach()
        self.logvars = self.logvars.detach()
        self.logprobs = (-0.5*self.logvars - (self.actions - self.means).pow(2) / 2*torch.exp(self.logvars))
        self.logprobs = torch.sum(self.logprobs, dim=1, keepdim=True)


    def data_generator(self, batch_size):
        """
        iterate over dataset with batch_size. Everything is detached.
        :return: a batch of data. An dictionary
        """
        sampler = BatchSampler(SubsetRandomSampler(range(self.total_size)), batch_size, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)

            batch = dict()
            batch['states'] = self.states[indices]
            batch['actions'] = self.actions[indices]
            batch['returns'] = self.returns[indices]
            batch['values'] = self.values[indices]
            batch['logprobs'] = self.logprobs[indices]

            yield batch
