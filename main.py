import torch
import torch.optim as optim
from dataclasses import dataclass
from typing import Literal, List
import loguru
import os
import time
from dataset import (
    AudioDataset,
    AudioDataLoader,
    MixedAudioDataset,
    MixedAudioDataLoader,
)
from DPTNet.others.optimizer_dptnet import TransformerOptimizer
from DPTNet.models import DPTNet_base
from DPTNet.solver import Solver


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class MainArgs:
    clean_dir: str
    dirty_dirs: List[str]
    target_dirs: List[str]
    batch_size: int
    sample_rate: int
    mode: Literal["train_and_eval", "train", "eval"]
    n: int = 64  # feature dim in DPT blocks
    w: int = 2  # filter length in encoder
    k: int = 250  # chunk size in frames
    d: int = 6  # number of DPT blocks
    h: int = 4  # number of hidden units in LSTM after multihead attention
    e: int = 256  # #channels before bottleneck
    use_cuda: bool = True
    epochs: int = 100
    max_norm: float = 5.0  # clip gradient norm at 5
    save_folder: str = "exp/temp"
    continue_from: str = ""
    start_epoch: int = 0
    warmup: bool = True
    model_path: str = "final.pth.tar"
    print_freq: int = 10
    checkpoint: bool = True


def main(args: MainArgs):
    # setup logging that prints and saves to a file
    logger = loguru.logger
    logger.add(
        os.path.join(
            args.save_folder,
            "{}-{}.log".format(args.mode, time.strftime("%Y%m%d-%H%M%S")),
        )
    )
    os.makedirs(args.save_folder, exist_ok=True)

    logger.info(args.save_folder)

    # data
    tr_dataset = MixedAudioDataset(args.clean_dir, args.dirty_dirs)
    cv_dataset = AudioDataset(
        alignment=args.w >> 1, dirty_folders=args.target_dirs, segment_len=None
    )
    tr_loader = MixedAudioDataLoader(
        alignment=args.w
        >> 1,  # ensure T devisible by W / 2, checkout DPTNet.models.Encoder for more details
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1)
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    model = DPTNet_base(
        enc_dim=args.e,
        feature_dim=args.n,
        hidden_dim=args.h,
        layer=args.d,
        segment_size=args.k,
        win_len=args.w,
    )
    logger.info(model)

    if args.use_cuda:
        model.cuda()
        model.to(device)

    optimizier = TransformerOptimizer(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        k=0.2,
        d_model=args.n,
        warmup_steps=4000,
        warmup=bool(args.warmup),
    )

    solver = Solver(data, model, optimizier, args)
    solver.run()


if __name__ == "__main__":
    main(
        MainArgs(
            clean_dir="./datasets/clean/pi/bootstrap/",
            dirty_dirs=["./datasets/dirty/c_chan/stardew_valley/"],
            target_dirs=["./datasets/dirty/pi/stardew_valley/"],
            batch_size=1,
            sample_rate=44100,
            mode="train_and_eval",
            save_folder="exp/test",
            w=16,  # for fast training & prototyping
            d=2,
        )
    )
