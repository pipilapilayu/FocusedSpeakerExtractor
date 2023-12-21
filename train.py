import torch
import torch.utils.data
from dataclasses import dataclass
from typing import List
from dataset import (
    MixedAudioDataset,
    MixedAudioDataLoader,
)
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from dptnet_modules import DPTNetModule, DPTNetModuleArgs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainArgs:
    clean_dir: str
    dirty_dirs: List[str]
    target_dirs: List[str]
    batch_size: int
    sample_rate: int
    module_args: DPTNetModuleArgs
    exp_name: str
    use_cuda: bool = True
    epochs: int = 100
    max_norm: float = 5.0  # clip gradient norm at 5
    start_epoch: int = 0
    warmup: bool = True
    model_path: str = "final.pth.tar"
    print_freq: int = 10
    checkpoint: bool = True


def train(args: TrainArgs):
    full_dataset = MixedAudioDataset(args.clean_dir, args.dirty_dirs)

    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )
    train_loader = MixedAudioDataLoader(
        alignment=args.module_args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = MixedAudioDataLoader(
        alignment=args.module_args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = DPTNetModule(args.module_args)
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
    )
    trainer.fit(model, train_loader, eval_loader)


if __name__ == "__main__":
    train(
        TrainArgs(
            clean_dir="./datasets/clean/pi/bootstrap/",
            dirty_dirs=["./datasets/dirty/c_chan/stardew_valley/"],
            target_dirs=["./datasets/dirty/pi/stardew_valley/"],
            batch_size=1,
            sample_rate=44100,
            module_args=DPTNetModuleArgs(
                w=16,  # for fast training & prototyping
                d=2,
            ),
            exp_name="test_lightning",
        )
    )
