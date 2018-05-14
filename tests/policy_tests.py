from src.utils import Runner
from src.storage import RollOut, Dataset
from src.envs import PendulumEnv
from src.models import MLPContinuousPolicy, MLPCritic

def test_rollout():
    env = PendulumEnv()
    actor = MLPContinuousPolicy(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0],
                                num_hidden=8)
    memory = RollOut()
    runner = Runner(env, memory, 100, 20, actor)
    runner.sample()

    t_states = memory.to_tensors()[0]
    means, logvars = actor(t_states)
    print(means.mean(), means.std())
    assert means.size(0) == 100
    assert means.size(1) == 1
    assert logvars.size(0) == 100
    assert logvars.size(1) == 1
    assert logvars[0].item() == logvars[1].item()

