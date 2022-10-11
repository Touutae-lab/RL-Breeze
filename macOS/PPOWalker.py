import enum
from email import policy
from sre_parse import State

import torch
import torch.nn as nn

from macOS.base.PPObase import PPOBase


class PPO(nn.Module):
    def __init__(
        self,
        env,
        policy_lr=1e-4,
    ):
        super().__init__()
        self.env = env
        self.policy = PPOBase()
        
        
        self.optim = nn.functional.AdamW(self.policy.parameters(), lr=policy_lr)

    def train(self, episode):
        transition = []
        for i in range():
            state = self.env.reset()
            ep_return = 0
            done = False
            while not done:
                action = self.policy(state).multinomial(1).detach()
                next_state, reward, done, _ = self.env.step()
                transition.append([state, action, ~done * reward])
                ep_turn += reward
                state = next_state
            for t, (state_t, action_t, reward_t) in reversed(
                list(enumerate(transition))
            ):
                G = reward_t + gamma * G
                prob_t = policy(state_t)
                log_prob_t = torch.log(prob_t + 1e-6)
                action_log_prob_t = log_prbos_t.gather(1, action_t)
