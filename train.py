import torch
import torch.utils.data
from dataclasses import dataclass
from typing import List
from dataset import (
    MixedAudioDataset,
    MixedAudioDataLoader,
    N2NMixedAudioDataset,
)
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from dptnet_modules import DPTNetModuleArgs, N2NDPTNetModule


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainArgs:
    clean_dir: str
    dirty_dirs: List[str]
    batch_size: int
    module_args: DPTNetModuleArgs
    exp_name: str
    epochs: int = 10


def train(args: TrainArgs):
    dataset = N2NMixedAudioDataset(MixedAudioDataset(args.clean_dir, args.dirty_dirs))

    loader = MixedAudioDataLoader(
        alignment=args.module_args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = N2NDPTNetModule(args.module_args)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath="exp/%s/checkpoints/" % args.exp_name,
        filename="model-{epoch:02d}",
    )
    trainer = Trainer(
        logger=TensorBoardLogger("tb_logs", name=args.exp_name),
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
    )
    trainer.fit(model, loader)


def train_and_eval(args: TrainArgs):
    full_dataset = MixedAudioDataset(args.clean_dir, args.dirty_dirs)

    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )
    n2n_train_dataset = N2NMixedAudioDataset(train_dataset)
    train_loader = MixedAudioDataLoader(
        alignment=args.module_args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=n2n_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = MixedAudioDataLoader(
        alignment=args.module_args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = N2NDPTNetModule(args.module_args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="exp/%s/checkpoints/" % args.exp_name,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    trainer = Trainer(
        logger=TensorBoardLogger("tb_logs", name=args.exp_name),
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
    )
    trainer.fit(model, train_loader, eval_loader)


if __name__ == "__main__":
    train_and_eval(
        TrainArgs(
            clean_dir="./datasets/clean/pi/bootstrap/",
            dirty_dirs=["./datasets/dirty/c_chan/stardew_valley/"],
            batch_size=2,
            module_args=DPTNetModuleArgs(
                d=2,
            ),
            exp_name="test_b2_d2_train_eval_n2n",
        )
    )
