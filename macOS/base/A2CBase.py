from turtle import forward
import torch
import torch.nn as nn

class ActorNet(nn.Module):
    def __init__(self, in_feature, out_dims, hidden_dims=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.Softmax(dim=-1)
        ) 
    
    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
        return