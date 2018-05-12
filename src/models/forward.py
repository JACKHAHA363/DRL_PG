"""
    Created by yuchen on 5/11/18
    Description: The class of forwarding model
"""
import math
import torch

class ForwardModel(object):
    """
    A base class for forward models
    """
    def __init__(self):
        pass

    def step(self, state, action):
        """
        The transition function. Can take batch.
        :param state: [bsz, state_dim]
        :param action_dim: [bsz, action_dim]
        :returns
        nextstate: [bsz, state_dim]
        reward: [bsz]
        """
        pass

class PendulumForward(ForwardModel):
    """
    The ground truth model for pendulum
    """
    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

    def step(self, state, action):
        """
        :param state: [bsz, 2]
        :param action: [bsz, 1]
        :return: nextstate [bsz, 2]. reward [bsz, 1]
        """
        # action [bsz]
        action = action.squeeze(-1)
        th = state[:, 0] # [bsz]
        thdot = state[:, 1] # [bsz]
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        # costs [bsz, 1]
        action = torch.clamp(action, -self.max_torque, self.max_torque)
        costs = th.pow(2) + 0.1 * thdot.pow(2) + 0.001 * action.pow(2)
        costs = costs.unsqueeze(-1)

        # [bsz]
        newthdot = thdot + (-3*g/(2*l) * torch.sin(th + math.pi) + 3./(m*l**2)*action)
        newth = th + newthdot*dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)

        # [bsz, 2]
        next_state = torch.stack([newth, newthdot], dim=1)
        return next_state, -costs

