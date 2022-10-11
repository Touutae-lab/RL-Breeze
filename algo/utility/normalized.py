import gym
import torch
from gym.wrappers import (NormalizeObservation, NormalizeReward,
                          RecordEpisodeStatistics, RecordVideo)


@torch.no_grad()
def test_agent(env, episode_length, policy, episodes=10):

    ep_returns = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            loc, scale = policy(state)
            sample = torch.normal(loc, scale)
            action = torch.tanh(sample)
            state, reward, done, info = env.step(action)
            ep_ret += reward.item()

        ep_returns.append(ep_ret)

    return sum(ep_returns) / episodes


def create_env(env_name, num_envs):
    env = gym.vector.make(env_name, num_envs=num_envs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    return env
