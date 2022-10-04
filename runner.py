import functools

import brax
import gym
import torch
from brax import envs
from brax.envs import to_torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Register environment from brax to gym
entry_points = functools.partial(envs.create_gym_env, env_name="ant")
gym.register("brax-ant", entry_point=entry_points)


# Create environment
env = gym.make("brax-ant", episode_length="1_000")
env = to_torch.JaxtoTorchWrapper(env, device="GPU")

create_video(env, 1_000)
