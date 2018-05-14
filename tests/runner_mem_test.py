from src.utils import Runner
from src.storage import RollOut, Dataset
from src.envs import PendulumEnv
from src.models import MLPContinuousPolicy, MLPCritic
import torch
import numpy as np

def test_rollout():
    env = PendulumEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=8)
    critic = MLPCritic(state_dim=env.observation_space.shape[0], num_hidden=10)
    memory = RollOut()
    runner = Runner(env, memory, 100, 20, actor)
    runner.sample()
    assert len(memory.states) == 100
    assert sum(memory.news) == 5

    dataset = Dataset(memory, actor, critic, 1)
    rets = dataset.returns
    assert np.allclose(rets[0].item(), torch.sum(dataset.rewards[0:20]).item())
    assert np.allclose(rets[20].item(), torch.sum(dataset.rewards[20:40]).item())
    assert np.allclose(rets[40].item(), torch.sum(dataset.rewards[40:60]).item())
    assert np.allclose(rets[60].item(), torch.sum(dataset.rewards[60:80]).item())



