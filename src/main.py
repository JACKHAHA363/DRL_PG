import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from envs import PendulumEnv
from models import MLPCritic, MLPContinuousPolicy
from storage import RollOut
from utils import sample_episode, compute_returns

def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--baseline', type=str, default=None,
                        help="[None]|[value]|[model]")
    parser.add_argument('--T', default=1000, type=int, help="maximum length of each episode")
    parser.add_argument('--steps', default=100, type=int, help="number of policy updates")
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    env = PendulumEnv()
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=10)
    critic = MLPCritic(state_dim=env.observation_space.shape[0], num_hidden=10)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()

    memory = RollOut()

    for step in range(args.steps):
        sample_episode(env, memory, actor, args.T, args.render)

        # preparing batcha
        states, actions, rewards, noise = memory.to_tensors()

        # train critic
        rets_np = compute_returns(memory, args.gamma)
        rets = torch.from_numpy(rets_np).unsqueeze(-1) # [T, 1]
        rets_pred = critic(states) # [T, 1]
        critic_loss = (rets - rets_pred).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        if args.baseline == 'value':
            baseline = rets_pred.detach()
        if args.baseline == 'model':
            raise NotImplementedError
        else:
            baseline = 0

        # train policy
        #logprobs = evaluate_actions(states, actions, actor)


if __name__ == '__main__':
    main()



