from src.envs import PendulumEnv
from src.models import PendulumForward
import torch
import ipdb
import numpy as np

env = PendulumEnv()
forward_model = PendulumForward()

# verify roll out
state_np = env.reset()
state = torch.from_numpy(state_np).unsqueeze(0)
for _ in range(10000):
    action = torch.rand([1,1], requires_grad=False) * 4 + 2
    action = action.double()
    action_np = action.numpy().reshape(1)
    next_state, reward = forward_model.step(state, action)
    next_state = next_state.squeeze(0)
    reward = reward.squeeze(0)
    next_state_np, reward_np, _, _ = env.step(action_np)
    assert np.allclose(next_state.numpy(), next_state)
    assert np.allclose(reward.numpy(), reward)

print("pass roll out test")

# gradient test
init_state = torch.from_numpy(env.reset()).unsqueeze(0)
init_state.requires_grad = True
state = init_state
rewards = 0
for _ in range(1):
    action = torch.rand([1,1]) * 4 + 2
    state, reward = forward_model.step(state, action.double())
    rewards += reward
rewards.backward()
grad_truth = [-2*init_state[0,0].item(), -2*0.1*init_state[0,1].item()]
assert np.allclose(np.array(grad_truth), init_state.grad.squeeze(0).numpy())
print("pass grad test")
