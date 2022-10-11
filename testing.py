from csv import writer
from math import gamma
from os import stat
from ssl import CertificateError

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.optim import AdamW


def actor_critic(actor, critic, episodes, alpha=1e-4, gamma=0.99):
    actor_optim = AdamW(actor.parameters(), lr=1e-3)
    critic_optim = AdamW(critic.parameters(), lr=1e-4)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}

    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        I = 1.0

        while not done_b.all():
            action = actor(state).multinomial(1).detach()
            next_state, reward, done, _ = parallel_env.step(action)

            value = critic(state)
            target = reward + ~done * gamma * critic(next_state).detach()
            critic_loss = F.mse_loss(value, target)
            critic.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            advantage = (target - value).detach()
            probs = actor(state)
            log_probs = torch.log(probs + 1e-6)
            action_log_prob = log_probs.gather(1, action)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
            actor_loss = -I * action_log_prob * advantage - 0.01 * entropy
            actor_loss = actor_loss.mean()
            actor.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            ep_return += reward
            done_b |= done
            state = next_state
            I = I * gamma

        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())

    return stats


env = gym.make("LunarLander-v2")
dims = env.observation_space.shape[0]
actions = env.action_space.n

actor = nn.Sequential(
    nn.Linear(dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, actions),
    nn.Softmax(dim=-1),
)

critic = nn.Sequential(
    nn.Linear(dims, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
)

actor_critic(actor, critic, episodes=200)
