import random

import numpy as np
from torch.utils.data.dataset import IterableDataset


# Deterministic Policy
class RLDataset(IterableDataset):
    def __init__(self, env, policy, steps_per_epoch, gamma) -> None:
        super().__init__()
        self.env = env
        self.policy = policy
        self.step_per_epoch = steps_per_epoch
        self.gamma = gamma

    def __iter__(self):
        transition = []

        for step in range(self.step_per_epoch):
            action = self.policy(self.obs)
            action = action.multinomial(1).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action.flattern())
            transition.append(self.obs, action, reward, done)
            self.obs = next_obs
        # Stack all item into tuple
        obs_b, action_b, reward_b, done_b = map(np.stack, zip(*transition))

        running_return = np.action(self.env.num_envs, dtype=np.float32)
        return_b = np.zeros_like(reward_b)

        # Calculate value in a single batch
        for row in range(self.step_per_epoch - 1, -1, -1):
            running_return = reward_b[row] + (
                1 - done_b[row] * self.gamma * running_return
            )
            return_b[row] = running_return
        # Make sure we have the right shape
        num_samples = self.env.nun_envs * self.steps_per_epoch
        obs_b = obs_b.reshape(num_samples, -1)
        action_b = action_b.reshape(num_samples, -1)
        return_b = return_b.reshape(num_samples, -1)

        # Shuffle transitions.
        index = list(range(num_samples))
        random.shuffle(index)

        for i in index:
            yield obs_b[i], action_b[i], return_b[i]
