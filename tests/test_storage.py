from src.storage import RollOut
from src.envs import PendulumEnv
from src.models import MLPContinuousPolicy
from src.utils import sample_episode

def test_rollout():
    env = PendulumEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=10)
    memory = RollOut()
    sample_episode(env, memory, T=50, actor=actor)
    states, actions, rewards, noise = memory.to_tensors()
    assert states.size(0) == 50
    assert actions.size(0) == 50
    assert rewards.size(0) == 50
    assert noise.size(0) == 50

    assert states.size(1) == 2
    assert actions.size(1) == 1
    assert rewards.size(1) == 1
    assert noise.size(1) == 1
