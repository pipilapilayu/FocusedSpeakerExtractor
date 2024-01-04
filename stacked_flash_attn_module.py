from dataclasses import dataclass
import itertools
from typing import Callable
import lightning
from flash_attn import flash_attn_qkvpacked_func
import torch
from torch import nn
from torchmetrics import ScaleInvariantSignalNoiseRatio
from dataset import MixedAudioDataLoaderOutput


@dataclass
class FlashAttentionBlockArgs:
    emb_dim: int
    num_heads: int
    dropout_p: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.num_heads


class FlashAttentionBlock(nn.Module):
    def __init__(self, args: FlashAttentionBlockArgs) -> None:
        super().__init__()

        assert (
            args.emb_dim % args.num_heads == 0
        ), "emb_dim must be divisible by num_heads"
        self.args = args

        self.qkv_linear = nn.Linear(args.emb_dim, args.emb_dim * 3)
        self.out_linear = nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, E] -> [B, S, E]"""
        batch_size, seq_len, emb_dim = x.shape
        qkv = self.qkv_linear(x).reshape(
            batch_size, seq_len, 3, self.args.num_heads, self.args.head_dim
        )  # [B, S, E] -> [B, S, 3 * E] -> [B, S, 3, #H, H]
        out = flash_attn_qkvpacked_func(
            qkv, dropout_p=self.args.dropout_p
        )  # [B, S, 3, #H, H] -> [B, S, #H, H]
        assert isinstance(
            out, torch.Tensor
        ), "attention must return a tensor"  # make pyright happy
        out = out.reshape(
            batch_size, seq_len, self.args.emb_dim
        )  # [B, S, #H, H] -> [B, S, E]
        return self.out_linear(out)


@dataclass
class MLPBlockArgs:
    io_size: int
    hidden_dim: int
    leaky_relu_neg_slope: float = 1e-2
    dropout_p: float = 0.5


class MLPBlock(nn.Module):
    def __init__(self, args: MLPBlockArgs) -> None:
        self.enc = nn.Conv1d(args.io_size, args.hidden_dim, 1)
        self.act = nn.LeakyReLU(args.leaky_relu_neg_slope)
        self.dec = nn.Conv1d(args.hidden_dim, args.io_size, 1)
        self.dropout = nn.Dropout(args.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, E] -> [B, S, E]"""
        return self.dropout(self.dec(self.act(self.enc(x))))


@dataclass
class FlashTransformerBlockArgs:
    attn_block_args: FlashAttentionBlockArgs
    mlp_block_args: MLPBlockArgs


class FlashTransformerBlock(nn.Module):
    def __init__(self, args: FlashTransformerBlockArgs) -> None:
        self.attn = FlashAttentionBlock(args.attn_block_args)
        self.norm = nn.LayerNorm(args.attn_block_args.emb_dim)
        self.mlp = MLPBlock(args.mlp_block_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, E] -> [B, S, E]"""
        norm_x = self.norm(x)
        attn_x = self.attn(norm_x)
        mlp_x = self.mlp(x)
        return mlp_x + attn_x + norm_x


@dataclass
class StackedCNNArgs:
    in_channels: int  # if input is B x 2 x T, in_channels is 2
    out_channels: int  # should be the same as `FlashAttentionBlockArgs.emb_dim`
    cnn_kernel_size: int  # must be odd
    cnn_layers: int


class StackedCNN(nn.Module):
    def __init__(self, args: StackedCNNArgs) -> None:
        assert args.cnn_layers >= 1, "must contain at least 1 layer of CNN"
        assert args.cnn_kernel_size & 1, "kernel size must be odd"
        self.cnns = nn.ModuleList(
            [
                nn.Conv1d(
                    args.in_channels,
                    args.out_channels,
                    args.cnn_kernel_size,
                    padding=args.cnn_kernel_size >> 1,
                )
            ]
            + [
                nn.Conv1d(
                    args.out_channels,
                    args.out_channels,
                    args.cnn_kernel_size,
                    padding=args.cnn_kernel_size >> 1,
                )
                for _ in range(1, args.cnn_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, I, T] -> [B, O, T]"""  # I, O means in_channels, out_channels
        for cnn in self.cnns:
            x = cnn(x)
        return x


@dataclass
class CoreSeparatorArgs:
    context_window: int
    encoder_args: StackedCNNArgs
    decoder_args: StackedCNNArgs
    ft_block_args: FlashTransformerBlockArgs
    num_ft_blocks: int


class CoreSeparator(nn.Module):
    def __init__(self, args: CoreSeparatorArgs) -> None:
        assert (
            args.encoder_args.out_channels == args.ft_block_args.attn_block_args.emb_dim
        ), "the output dim of encoder must be the same as hidden dim of separator"
        assert (
            args.decoder_args.in_channels == args.ft_block_args.attn_block_args.emb_dim
        ), "the input dim of decoder must be the same as hidden dim of separator"
        assert (
            args.encoder_args.in_channels == args.decoder_args.out_channels
        ), "the number of input and output channel must be the same"
        assert (
            args.context_window > 0
            and (args.context_window & -args.context_window) == args.context_window
        ), "the size of context window must be power of two"

        self.encoder = StackedCNN(args.encoder_args)
        self.separator = nn.ModuleList(
            [
                FlashTransformerBlock(args.ft_block_args)
                for _ in range(args.num_ft_blocks)
            ]
        )
        self.decoder = StackedCNN(args.decoder_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.separator(self.encoder(x)))


# N2C module
class SeparatorModule(lightning.LightningModule):
    def __init__(self, args: CoreSeparatorArgs) -> None:
        self.model = CoreSeparator(args)
        self.segment_size = args.context_window  # might add hop size later, not for now
        self.loss = ScaleInvariantSignalNoiseRatio()

    @staticmethod
    def _apply_on_blocks(
        segment_size: int, x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        batch_size, num_channel, x_len = x.shape

        pad_x_len = x_len + (segment_size - x_len % segment_size) % segment_size
        padded_x = torch.zeros(*x.shape[:-1], pad_x_len)
        padded_x[..., :x_len] = x

        segmented_x = padded_x.reshape(batch_size, num_channel, -1, segment_size)
        estimated_y = torch.zeros_like(padded_x)
        for i in range(pad_x_len // segment_size):
            estimated_y[..., i, :] = fn(segmented_x[..., i, :].squeeze()).unsqueeze(
                dim=-2
            )

        return estimated_y.reshape(batch_size, num_channel, -1)[..., :x_len]

    def _shared_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        mixed, clean = batch
        return 0 - self.loss(
            self._apply_on_blocks(self.segment_size, mixed, self.model), clean
        )

    def _shared_log_step(
        self, batch: MixedAudioDataLoaderOutput, loss_name: str
    ) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log(loss_name, loss)
        return loss

    def training_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        return self._shared_log_step(batch, "train_loss")

    def validation_step(self, batch: MixedAudioDataLoaderOutput) -> torch.Tensor:
        return self._shared_log_step(batch, "val_loss")
