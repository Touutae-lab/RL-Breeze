import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .PolicyGradient import PolicyGradient
from .utility.normalized import create_env
from .utility.RLDataset import RLDataset


class ReinForce(LightningModule):
    def __init__(
        self,
        env_name,
        num_envs=8,
        samples_per_epoch=1000,
        batch_size=1024,
        hidden_size=64,
        policy_lr=0.001,
        gamma=0.99,
        entropy_coef=0.001,
        optim=AdamW,
    ):

        super().__init__()

        self.env = create_env(env_name, num_envs=num_envs)

        obs_size = self.env.single_observation_space.shape[0]
        n_actions = self.env.single_action_space.n

        self.policy = PolicyGradient(obs_size, n_actions, hidden_size)
        self.dataset = RLDataset(self.env, self.policy, samples_per_epoch, gamma)

        self.save_hyperparameters()

    # Configure optimizers.
    def configure_optimizers(self):
        return self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    # Training step.
    def training_step(self, batch, batch_idx):
        obs, actions, returns = batch

        probs = self.policy(obs)
        log_probs = torch.log(probs + 1e-6)
        action_log_prob = log_probs.gather(1, actions)

        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

        pg_loss = -action_log_prob * returns
        loss = (pg_loss - self.hparams.entropy_coef * entropy).mean()

        self.log("episode/PG Loss", pg_loss.mean())
        self.log("episode/Entropy", entropy.mean())

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("episode/Return", self.env.return_queue[-1])
