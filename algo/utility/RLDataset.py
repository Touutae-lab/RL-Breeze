import random

import torch
import numpy as np
from torch.utils.data.dataset import IterableDataset

# Deterministic Policy

class RLDataset(IterableDataset):

  def __init__(self, env, policy, steps_per_epoch, gamma):
    self.env = env
    self.policy = policy
    self.steps_per_epoch = steps_per_epoch
    self.gamma = gamma
    self.obs = env.reset()

  @torch.no_grad()
  def __iter__(self):
    transitions = []

    for step in range(self.steps_per_epoch):
      action = self.policy(self.obs)
      action = action.multinomial(1).cpu().numpy()
      next_obs, reward, done, info = self.env.step(action.flatten())
      transitions.append((self.obs, action, reward, done))
      self.obs = next_obs

    obs_b, action_b, reward_b, done_b = map(np.stack, zip(*transitions))
    
    running_return = np.zeros(self.env.num_envs, dtype=np.float32)
    return_b = np.zeros_like(reward_b)

    for row in range(self.steps_per_epoch - 1, -1, -1):
      running_return = reward_b[row] + (1 - done_b[row]) * self.gamma * running_return
      return_b[row] = running_return

    num_samples = self.env.num_envs * self.steps_per_epoch
    obs_b = obs_b.reshape(num_samples, -1)
    action_b = action_b.reshape(num_samples, -1)
    return_b = return_b.reshape(num_samples, -1)

    idx = list(range(num_samples))
    random.shuffle(idx)

    for i in idx:
      yield obs_b[i], action_b[i], return_b[i]
