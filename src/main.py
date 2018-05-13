import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# get around pytest. should be fixed in the future
if __name__ == '__main__':
    from envs import PendulumEnv
    from models import MLPCritic, MLPContinuousPolicy
    from storage import RollOut
else:
    from .envs import PendulumEnv
    from .models import MLPCritic, MLPContinuousPolicy
    from .storage import RollOut


def sample_episode(env, memory, actor, T, render=False):
    """
    sample from `env` and store things into `memory` from `actor`
    """
    noise = torch.zeros([1,1])
    state = env.reset()
    memory.reset()
    t = 0
    while t < T:
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        mean, logvar = actor(state_t)
        noise.normal_()
        action_t = mean + noise * torch.exp(0.5 * logvar)
        action = action_t.detach().numpy()[0]

        # env step
        next_state, reward, done, _ = env.step(action)
        memory.add_transition(state, action, reward, noise.item())
        state = next_state
        t += 1

        if render:
            env.render()
        if done:
            break

def compute_returns(memory, gamma):
    """
    Compute the returns
    :return: [num_states]. np.array
    """
    rewards = memory.rewards
    assert len(rewards) > 0 # make it's not empty
    returns = np.zeros([len(rewards)])
    ret = 0
    for i in reversed(range(len(rewards))):
        returns[i] = rewards[i] + gamma * ret
        ret = returns[i]
    return returns


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


#########MAIN#################################################
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
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    main()



