from argparse import Action

import torch
import torch.nn as nn


class PPOBase(nn.Module):
    def __init__(self, in_features, out_dim, hidden_layer=512):
        super().__init__()
        self.network = nn.Sequantial(
            nn.Linear(in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, int(hidden_layer / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_layer / 2), out_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.network(x)
