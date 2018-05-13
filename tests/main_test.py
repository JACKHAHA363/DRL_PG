from src.main import sample_episode, compute_returns
from src.envs import PendulumEnv, Continuous_MountainCarEnv
from src.models import MLPContinuousPolicy
from src.storage import RollOut


def test_sample_episodes():
    env = PendulumEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=10)
    memory = RollOut()
    sample_episode(env, memory, actor, 50, render=True)
    assert len(memory.states) == 50


def test_sample_episodes_mc():
    env = Continuous_MountainCarEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=10)
    memory = RollOut()
    sample_episode(env, memory, actor, 50, render=True)
    assert len(memory.states) == 50


def test_compute_returns():
    env = Continuous_MountainCarEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=10)
    memory = RollOut()
    sample_episode(env, memory, actor, 100, render=True)
    returns = compute_returns(memory, gamma=1)
    assert returns[-1] == memory.rewards[-1]
    assert returns[0] == sum(memory.rewards)
