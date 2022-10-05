import gym
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch.optim import AdamW


class PPO(LightningModule):
    def __init__(
        self,
        batch_size=1024,
        hidden_size=256,
        sample_per_epoch=5,
        epoch_repeat=8,
        policy_lr=1e-3,
        value_lr=1e-4,
        gamma=0.97,
        epsilon=0.3,
        entropy_coef=0.1,
        optimizer=AdamW,
    ) -> None:
        super().__init__()
        
        self.policy
        self.value
        self.target_value
        
        self.obs_size
        self.action
        
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters() lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=slef.hparams.policy_lr)
        return value_opt, policy_opt
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)
    
    def training_step(self, batch, batch_idx):
        return 