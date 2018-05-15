import torch


class Runner(object):
    """
    Sample from env and store it into memory
    """
    def __init__(self, env, memory, batch_size, T, actor):
        self.env = env
        self.memory = memory
        self.batch_size = batch_size
        self.T = T
        self.actor = actor
        self.stats = {} # empty dict

    def sample(self, render=False):
        """
        Sample a batch of transitions from envs.
        Reporting statistics as well.
        :return stats. performance summary about the batch
        """
        self.memory.reset()
        nb_transitions = 0

        total_rewards = 0
        num_episodes = 0
        while nb_transitions < self.batch_size:
            state = self.env.reset()
            num_episodes += 1
            episode_length = 0
            while episode_length < self.T:
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                dist = self.actor(state_t)
                action = dist.sample()
                logprob = dist.log_prob(action).item()
                noise = ((action - dist.mean) / dist.stddev).item()

                # env step
                action = action.numpy()[0]
                next_state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()

                episode_length += 1
                nb_transitions += 1
                total_rewards += reward

                done = done or episode_length == self.T
                new = 1 if done else 0
                self.memory.add_transition(state, action, logprob,
                                           reward, noise, new)
                state = next_state

                if done:
                    break


        self.stats['average_reward'] = total_rewards / self.batch_size
        self.stats['num_episodes'] = num_episodes
        return self.stats


