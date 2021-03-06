"""
    Created by yuchen on 5/9/18
    Description: Policies network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MLPCritic(nn.Module):
    """
    For classic control
    """
    def __init__(self, state_dim, num_hidden=10):
        """
        :param state_dim: state dimension
        :param num_hidden: number of hidden units
        """

        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1),
        )

        self.apply(self.weight_init)

    def forward(self, states):
        """
        :param states: [bsz, state_dim]
        :return: state value. [bsz]
        """
        return self.main(states)

    def weight_init(self, m):
        """
        weight initialization
        """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


class MLPContinuousPolicy(nn.Module):
    """
    For classic control
    """
    def __init__(self, state_dim, action_dim, num_hidden=20):
        """
        :param state_dim: state dimension
        :param action_dim: number of actions
        :param num_hidden: number of hidden units
        """

        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(num_hidden, action_dim)
        self.logvars = nn.Parameter(0 * torch.ones(1, action_dim))

        self.apply(self.weight_init)

    def forward(self, states):
        """
        :param states: [bsz, state_dim]
        :return: a Normal distributionmeans [bsz, action_dim]. logvars [bsz, action_dim]
        """
        tmp = self.base(states)
        means = self.mean_head(tmp)
        logvars = self.logvars.expand_as(means)
        std = torch.exp(0.5 * logvars)
        return Normal(means, std)

    def weight_init(self, m):
        """
        weight initialization
        """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


class MLPDiscretePolicy(nn.Module):
    """
    For classic control
    """
    def __init__(self, state_dim, action_dim, num_hidden=10):
        """
        :param state_dim: state dimension
        :param action_dim: number of actions
        :param num_hidden: number of hidden units
        """

        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, num_hidden),
            nn.SELU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.SELU(inplace=True),
            nn.Linear(num_hidden, action_dim)
        )

    def forward(self, states):
        """
        :param states: [bsz, state_dim]
        :return: probability over action. [bsz, action_dim]
        """
        logits = self.main(states)
        return F.softmax(logits, dim=1)
