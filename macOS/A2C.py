from csv import writer
from math import gamma
from os import stat
from ssl import CertificateError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange

from macOS.base.A2CBase import Actor, ValueNet


class A2C:
    def __init__(
        self,
        env,
        writer,
        hidden_dim=512,
        policy_lr=1e-3,
        value_lr=1e-3,
        discount_y=0.99,
    ):
        super().__init__()
        self.discout = discount_y
        self.env = env
        self.writer = writer
        self.policy = Actor(in_feature, out_dims, hidden_dim)
        self.value_net = ValueNet(in_feature, out_dims, hidden_dim)

        self.policy_optim = AdamW(self.policy.parameters(), lr=policy_lr)
        self.value_optim = AdamW(self.value_net, lr=value_lr)

    def learn(self, epoches):
        for epoch in trange(epoches):
            state = self.env.reset()
            done = True
            while not done:
                action = self.policy(state).multinomial(1).detach()

                next_state, reward, done, _ = self.env.step(action)

                value = self.value_net(state)
                target = (
                    reward + ~done * self.discout * self.value_net(next_state).detach()
                )

                critic_loss = F.mse(value, target)

                self.value_net.zero_grad()
                critic_loss.backward()

                self.value_optim.step()

                advantage = (target - value).detach()
                probs = self.policy(state)
                log_probs = torch.log(probs + 1e-6)
                action_log_prob = log_probs.gather(1, action)
                entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
                actor_loss = -I * action_log_prob * advantage - 0.01 * entropy
                actor_loss = actor_loss.mean()

                self.policy.zero_grad()
                actor_loss.backward()
                self.policy_optim.step()

                ep_return += reward

                state = next_state
                I = I * self.discout
            self.writer.add_scalar("Actor loss/Epoch", actor_loss.item(), epoch)
            self.writer.add_scalar("Critic Loss/Epoch", critic_loss.item(), epoch)
            self.writer.add_scalar("Reward/Epoch", ep_return.mean().item(), epoch)
        return True
