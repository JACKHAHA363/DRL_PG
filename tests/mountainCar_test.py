from src.envs import Continuous_MountainCarEnv
from src.models import ContinuousMountainCarForward
import torch
import numpy as np

env = Continuous_MountainCarEnv()
forward_model = ContinuousMountainCarForward()

# verify roll out
def test_rollout():
    state_np = env.reset()
    state = torch.from_numpy(state_np).unsqueeze(0)
    for _ in range(10000):
        action = torch.rand([1,1], requires_grad=False) * 2 - 1
        action = action.double()
        action_np = action.numpy().reshape(1)
        next_state, reward = forward_model.step(state, action)
        next_state = next_state.squeeze(0)
        reward = reward.squeeze(0)
        next_state_np, reward_np, _, _ = env.step(action_np)
        assert np.allclose(next_state.numpy(), next_state)
        assert np.allclose(reward.numpy(), reward)


def test_gradient():
    # gradient test
    init_state = torch.from_numpy(env.reset()).unsqueeze(0)
    init_state.requires_grad = True
    state = init_state
    rewards = 0
    for _ in range(1):
        action = torch.rand([1,1]) * 2 - 2
        action = action.double()
        state, reward = forward_model.step(state, action)
        rewards += reward
    rewards.backward()
    pos = init_state[0, 0].item()
    vel = init_state[0, 1].item()
    grad_truth = [1 + 3*0.0025*np.sin(3*pos), 1]
    assert np.allclose(np.array(grad_truth), init_state.grad.squeeze(0).numpy())
    print("pass grad test")


def test_input_output():
    state = torch.rand([20, 2])
    action = torch.rand([20, 1])
    nextstate, reward = forward_model.step(state, action)
    assert nextstate.size(0) == 20
    assert nextstate.size(1) == 2
    assert reward.size(0) == 20
    assert reward.size(1) == 1
