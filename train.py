import argparse
import os

from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter

from algo.ReinForce import ReinForce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym", type=str, default="LunarLander-v2")
    parser.add_argument("--lr", type=str, default=1e-3)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--wandb", type=str, default="RL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--algo", type=str, default="Rainforce")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    run_name = f"{args.gym}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter("run/")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    algo = ReinForce(args.gym)
    trainer = Trainer(max_epochs=args.step)
    trainer.fit(algo)
