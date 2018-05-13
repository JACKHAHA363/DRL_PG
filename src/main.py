import argparse


import torch
import torch.optim as optim

from envs import PendulumEnv, Continuous_MountainCarEnv
from models import MLPCritic, MLPContinuousPolicy
from storage import RollOut
from utils import sample_episode, compute_returns
from tensorboardX import SummaryWriter

def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env', type=str, default='pendulum',
                        help='[pendulum] | [mountaincar]')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--baseline', type=str, default=None,
                        help="[None]|[value]|[model]")
    parser.add_argument('--T', default=200, type=int, help="maximum length of each episode")
    parser.add_argument('--steps', default=10000, type=int, help="number of policy updates")
    parser.add_argument('--beta', default=0.001, type=float,
                        help='the weight of KL term')
    args = parser.parse_args()
    return args


def main():
    args = parser_args()

    if args.env == 'pendulum':
        from gym.envs.classic_control import PendulumEnv
        env = PendulumEnv()
    elif args.env == 'mountaincar':
        from gym.envs.classic_control import Continuous_MountainCarEnv
        env = Continuous_MountainCarEnv()
    else:
        raise NotImplementedError

    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=30)
    critic = MLPCritic(state_dim=env.observation_space.shape[0], num_hidden=30)
    actor_opt = optim.Adam(actor.parameters(), lr=1e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-5)

    memory = RollOut()

    writer = SummaryWriter(log_dir='{args.env}_{args.baseline}'.format(args=args))
    for step in range(args.steps):
        sample_episode(env, memory, actor, args.T, args.render)

        # preparing batcha
        states, actions, rewards, noise = memory.to_tensors()

        # train critic
        rets_np = compute_returns(memory, args.gamma)
        rets = torch.from_numpy(rets_np).float().unsqueeze(-1) # [T, 1]
        rets_pred = critic(states) # [T, 1]
        critic_loss = (rets - rets_pred).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # baseline [bsz, 1]
        if args.baseline == 'value':
            baseline = rets_pred.detach()
        elif args.baseline == 'model':
            raise NotImplementedError
        else:
            baseline = 0

        # reinforce logprobs with returns/advantages
        # another forward pass to evaluate actions
        means, logvars = actor(states) # [T, action_dim]

        # logprobs = -0.5 logvar - (x-mu)^2/2sigma^2
        vars = torch.exp(logvars)

        # [bsz, 1]
        logprobs = (-0.5*logvars - (actions - means).pow(2) / 2*vars)
        logprobs = torch.sum(logprobs, dim=1, keepdim=True)
        actor_loss = -((rets - baseline) * logprobs).mean()

        # add correction bias term
        if args.baseline == 'world':
            raise NotImplementedError

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # record statistics
        if (step+1) % args.log_interval == 0:
            print("at step {}\treward per step{:.2f}\t".format(
                step+1, rewards.mean().item())
            )
        writer.add_scalar('reward_per_step', rewards.mean().item(), step+1)

    print("display testing")
    sample_episode(env, memory, actor, args.T, render=True)

if __name__ == '__main__':
    main()



