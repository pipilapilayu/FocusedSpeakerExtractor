from dataclasses import dataclass
import lightning
from DPTNet.models import DPTNet_base
from DPTNet.others.pit_criterion import cal_loss
import torch
from dataset import MixedAudioDataLoaderOutput
from typing import Dict, Any
from torch.optim.lr_scheduler import ExponentialLR


def rms_loudness(signal: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(signal**2, dim=-1, keepdim=True))


def loudness_loss(
    estimated_signal: torch.Tensor, target_signal: torch.Tensor
) -> torch.Tensor:
    estimated_loudness = rms_loudness(estimated_signal)
    target_loudness = rms_loudness(target_signal)
    return (
        torch.abs(estimated_loudness - target_loudness) * 128
    )  # 100 is loudness_loss weight


@dataclass
class DPTNetModuleArgs:
    n: int = 64  # feature dim in DPT blocks
    w: int = 2  # filter length in encoder
    k: int = 250  # chunk size in frames
    d: int = 6  # number of DPT blocks
    h: int = 4  # number of hidden units in LSTM after multihead attention
    e: int = 256  # #channels before bottleneck
    warmup: bool = True


class DPTNetModule(lightning.LightningModule):
    def __init__(self, args: DPTNetModuleArgs):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = DPTNet_base(
            enc_dim=args.e,
            feature_dim=args.n,
            hidden_dim=args.h,
            layer=args.d,
            segment_size=args.k,
            win_len=args.w,
        )

    def _shared_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        padded_mixture, mixture_lengths, padded_source = batch

        estimate_source = self.model(padded_mixture)
        si_snr_loss = cal_loss(padded_source, estimate_source, mixture_lengths)
        rms_loss = loudness_loss(estimate_source, padded_source)
        loss = si_snr_loss + rms_loss

        return loss

    def training_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        return self._shared_step(batch)

    def validation_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        return self._shared_step(batch)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.001
        )

        # Exponential decay phase
        decay_factor = 0.98**0.5  # 0.98 every two epoch
        exp_scheduler = ExponentialLR(optimizer, gamma=decay_factor)

        return {
            "optimizer": optimizer,
            "lr_scheduler": exp_scheduler,
        }
