import torch
import numpy as np

def sample_episode(env, memory, actor, T, render=False):
    """
    sample from `env` and store things into `memory` from `actor`
    :param env: `env` object
    :param memory: rollout object
    :param actor: policy network
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
    :param memory: rollout object
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