import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyGradient(nn.Module):
    def __init__(self, in_features, n_action, hidden_size=512, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_action)

    def forward(self, x):
        x = torch.tensor(x).float().to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # [ [x1, x2, x3], [y1, y2, y3] ]
        return x
