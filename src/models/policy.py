"""
    Created by yuchen on 5/9/18
    Description: Policies network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.SELU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.SELU(inplace=True),
            nn.Linear(num_hidden, 1),
        )

    def forward(self, states):
        """
        :param states: [bsz, state_dim]
        :return: state value. [bsz]
        """
        return self.main(states)


class MLPContinuousPolicy(nn.Module):
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
        self.base = nn.Sequential(
            nn.Linear(state_dim, num_hidden),
            nn.SELU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.SELU(inplace=True)
        )
        self.mean_head = nn.Linear(num_hidden, action_dim)
        self.logvar_head = nn.Linear(num_hidden, action_dim)

    def forward(self, states):
        """
        :param states: [bsz, state_dim]
        :return: means [bsz, action_dim]. logvars [bsz, action_dim]
        """
        tmp = self.base(states)
        return self.mean_head(tmp), self.logvar_head(tmp)


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
