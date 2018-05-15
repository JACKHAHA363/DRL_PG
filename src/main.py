import argparse


import torch
import torch.optim as optim

from envs import PendulumEnv, Continuous_MountainCarEnv
from models import MLPCritic, MLPContinuousPolicy
from storage import RollOut, Dataset
from utils import Runner
from tensorboardX import SummaryWriter

def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

    parser.add_argument('--env', type=str, default='pendulum',
                        help='[pendulum] | [mountaincar]')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log_interval', type=int, default=2, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--baseline', type=str, default=None,
                        help="[None]|[value]|[model]")
    parser.add_argument('--T', default=100, type=int, help="maximum length of each episode")
    parser.add_argument('--batch_size', default=10000, type=int,
                        help='number of transitions each iteration')
    parser.add_argument('--steps', default=10000, type=int, help="number of policy updates")

    # policy optimization
    parser.add_argument('--actor_epochs', default=1, type=int)
    parser.add_argument('--actor_bsz', default=32, type=int)

    # critic optimization
    parser.add_argument('--critic_epochs', default=1, type=int)
    parser.add_argument('--critic_bsz', default=32, type=int)

    # ppo
    parser.add_argument('--kl_coef', default=0.1, type=float)

    # entropy
    parser.add_argument('--ent_coef', default=0.01, type=float)

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
                                num_hidden=10)
    critic = MLPCritic(state_dim=env.observation_space.shape[0], num_hidden=10)
    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    memory = RollOut()

    writer = SummaryWriter(log_dir='{args.env}_{args.baseline}'.format(args=args))
    runner = Runner(env, memory, args.batch_size, args.T, actor)
    for step in range(args.steps):
        # preparing dataset
        runner.sample()
        dataset = Dataset(memory, actor, critic, args.gamma)

        # train critic
        for _ in range(args.critic_epochs):
            critic_data_gen = dataset.data_generator(args.critic_bsz)
            for batch in critic_data_gen:
                # a forward pass to get value
                values = critic(batch['states'])
                targets = batch['returns']
                critic_loss = (targets - values).pow(2).mean()
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

        # train policy
        for _ in range(args.actor_epochs):
            actor_data_gen = dataset.data_generator(args.actor_bsz)
            for batch in actor_data_gen:
                rets = batch['returns']
                if args.baseline == 'value':
                    baseline = batch['values']
                elif args.baseline == 'model':
                    # add mprop stuff here
                    raise NotImplementedError
                elif args.baseline is None:
                    baseline = 0
                else:
                    raise NotImplementedError

                # reinforce logprobs with returns/advantages
                # another forward pass to evaluate actions
                states = batch['states']
                actions = batch['actions']
                old_logprobs = batch['logprobs']
                curr_dist = actor(states) # [T, action_dim]
                curr_logprobs = curr_dist.log_prob(actions)

                # surrogate loss
                surrogate_loss = - torch.exp(curr_logprobs - old_logprobs) * (rets - baseline)
                surrogate_loss = torch.mean(surrogate_loss)

                # add correction bias term
                if args.baseline == 'world':
                    raise NotImplementedError

                # add entropy term
                entropy = curr_dist.entropy().mean()
                surrogate_loss -= args.ent_coef * entropy

                actor_opt.zero_grad()
                surrogate_loss.backward()
                actor_opt.step()

        # record statistics
        if (step+1) % args.log_interval == 0:
            stats = runner.stats
            print('------- iter{} ----------'.format(step+1))
            for key in stats.keys():
                print("{} : {:.3f}".format(key, stats[key]))
                writer.add_scalar(key, stats[key], step+1)


if __name__ == '__main__':
    main()



