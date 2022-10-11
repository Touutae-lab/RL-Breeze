
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from algo.PolicyGradient import PolicyGradient


class PPO(nn.Module):

    def __init__(self, env_name, num_envs=2048, episode_length=1000,
                batch_size=1024, hidden_size=256, samples_per_epoch=5,
                epoch_repeat=8, policy_lr=1e-4, value_lr=1e-3, gamma=0.97,
                epsilon=0.3, entropy_coef=0.1, optim=AdamW):

    super().__init__()
