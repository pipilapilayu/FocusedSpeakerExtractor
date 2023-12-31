from typing import Callable, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import overlap_and_add
from torch.autograd import Variable
from .transformer_improved import TransformerEncoderLayer

EPS = 1e-8


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""

    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        # Components
        # 50% overlap
        self.conv2d_U = nn.Conv2d(
            1, N, kernel_size=(1, W), stride=(1, W // 2), bias=False
        )

    def forward(self, mixture):
        """
        Args:
            mixture: [B, 2, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, 2, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = torch.unsqueeze(mixture, 1)  # [B, 1, 2, T]
        mixture_w = F.relu(self.conv2d_U(mixture))  # [B, N, 2, L]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [B, E, L]
            est_mask: [B, E, L]
        Returns:
            est_source: [B, T]
        """
        # D = W * M
        # print(mixture_w.shape)
        # print(est_mask.shape)
        source_w = mixture_w * est_mask  # [B E 2 L]
        source_w = source_w.permute(0, 2, 3, 1)
        # S = DV
        est_source = self.basis_signals(source_w)  # [B 2 L W]
        est_source = overlap_and_add(est_source, self.W // 2)  # [B 2 T]
        return est_source


class SingleTransformer(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """

    def __init__(self, input_size, hidden_size, dropout):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,
            hidden_size=hidden_size,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
        )

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        transformer_output = (
            self.transformer(output.permute(1, 0, 2).contiguous())
            .permute(1, 0, 2)
            .contiguous()
        )
        return transformer_output


@contextmanager
def temporary_permute(tensor: torch.Tensor, dims: Tuple[int, ...]):
    inv_dims = [0] * len(dims)
    for i, v in enumerate(dims):
        inv_dims[v] = i

    permuted_tensor = tensor.permute(*dims).contiguous()
    try:
        yield permuted_tensor
    finally:
        tensor.copy_(permuted_tensor.permute(*inv_dims))


def as_permute(
    tensor: torch.Tensor,
    dims: Tuple[int, ...],
    f: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    inv_dims = [0] * len(dims)
    for i, v in enumerate(dims):
        inv_dims[v] = i

    return f(tensor.permute(*dims).contiguous()).permute(*inv_dims).contiguous()


# dual-path transformer
class DPT(nn.Module):
    """
    Deep dual-path transformer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        num_layers: int, number of stacked Transformer layers. Default is 1.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(DPT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path transformer
        self.row_transformers = nn.ModuleList([])
        self.col_transformers = nn.ModuleList([])
        self.channel_transformers = nn.ModuleList([])
        for _ in range(num_layers):
            self.row_transformers.append(
                SingleTransformer(input_size, hidden_size, dropout)
            )
            self.col_transformers.append(
                SingleTransformer(input_size, hidden_size, dropout)
            )
            self.channel_transformers.append(
                SingleTransformer(input_size, hidden_size, dropout)
            )

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(input_size, output_size, 1))

    def forward(self, input: torch.Tensor):
        """
        apply transformer on segment_len dim first and then num_segment dim, finally inter-channel
        Args:
            input: B x N x 2 x segment_len x num_segment
        Returns:
            B x self.output_size x 2 x segment_len x num_segment
        """

        batch_size, input_size, num_channel, dim1, dim2 = input.shape
        output = input
        for row_model, col_model, ch_model in zip(
            self.row_transformers, self.col_transformers, self.channel_transformers
        ):
            with temporary_permute(output, (0, 4, 2, 3, 1)) as permuted_input:
                row_input = permuted_input.view(
                    batch_size * dim2 * num_channel, dim1, input_size
                )
                row_output = row_model(row_input)
                permuted_input.copy_(
                    row_output.view(batch_size, dim2, num_channel, dim1, input_size)
                )

            with temporary_permute(output, (0, 3, 2, 4, 1)) as permuted_input:
                col_input = permuted_input.view(
                    batch_size * dim1 * num_channel, dim2, input_size
                )
                col_output = col_model(col_input)
                permuted_input.copy_(
                    col_output.view(batch_size, dim1, num_channel, dim2, input_size)
                )

            # after DPT, we add inter-channel attention, making it TPT
            with temporary_permute(output, (0, 3, 4, 2, 1)) as permuted_input:
                ch_input = permuted_input.view(
                    batch_size * dim1 * dim2, num_channel, input_size
                )
                ch_output = ch_model(ch_input)
                permuted_input.copy_(
                    ch_output.view(batch_size, dim1, dim2, num_channel, input_size)
                )

        output = output.view(batch_size, input_size, num_channel * dim1 * dim2)
        output = self.output(output)  # B, output_size, num_channel * dim1 * dim2
        output = output.view(batch_size, input_size, num_channel, dim1, dim2)

        return output


# base module for deep DPT
class DPT_base(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        layer=6,
        segment_size=250,
    ):
        super(DPT_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv2d(self.input_dim, self.feature_dim, (1, 1), bias=False)

        # DPT model
        self.DPT = DPT(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            num_layers=layer,
        )

    def pad_segment(
        self, input: torch.Tensor, segment_size: int
    ) -> Tuple[torch.Tensor, int]:
        # input is the features: (B, N, 2, T)
        batch_size, dim, num_channel, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, num_channel, rest)).type(
                input.type()
            )
            input = torch.cat([input, pad], 3)

        pad_aux = Variable(
            torch.zeros(batch_size, dim, num_channel, segment_stride)
        ).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 3)

        return input, rest

    def split_feature(self, input: torch.Tensor, segment_size: int):
        """
        split the feature into chunks of segment size
        Args:
            input: B x N x 2 x T
        Returns:
            B x N x 2 x segment_size x num_segment
        """

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, num_channel, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[..., :-segment_stride]
            .contiguous()
            .view(batch_size, dim, num_channel, -1, segment_size)
        )
        segments2 = (
            input[..., segment_stride:]
            .contiguous()
            .view(batch_size, dim, num_channel, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, num_channel, -1, segment_size)
            .transpose(-2, -1)
        )

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: [B N 2 L K]

        batch_size, dim, num_channel, segment_size, num_segment = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(-2, -1)
            .contiguous()
            .view(batch_size, dim, num_channel, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[..., :segment_size]
            .contiguous()
            .view(batch_size, dim, num_channel, -1)[..., segment_stride:]
        )
        input2 = (
            input[..., segment_size:]
            .contiguous()
            .view(batch_size, dim, num_channel, -1)[..., :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[..., :-rest]

        return output.contiguous()  # B, N, 2, T

    def forward(self, input):
        pass


class BF_module(DPT_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid()
        )

    def forward(self, input):
        # input = input.to(device)
        # input: (B, E, 2, T)
        batch_size, E, num_channel, seq_length = input.shape

        enc_feature = self.BN(input)  # [B E 2 L]-->[B N 2 L]
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(
            enc_feature, self.segment_size
        )  # [B N 2 L K]: L is the segment_size
        # print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPT
        output = self.DPT(enc_segments).view(
            batch_size, self.feature_dim, num_channel, self.segment_size, -1
        )  # [B N 2 L K]

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # [B N 2 T]

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # [B N 2 T]
        # bf_filter = (
        #     bf_filter.permute(0, 2, 3, 1)
        #     .contiguous()
        #     .view(batch_size, num_channel, -1, self.feature_dim)
        # )  # [B 2 T N]

        return bf_filter


# base module for DPTNet_base
class DPTNet_base(nn.Module):
    def __init__(
        self,
        enc_dim: int,
        feature_dim,
        hidden_dim,
        layer: int,
        win_len: int,
        segment_size=250,
    ):
        super(DPTNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.eps = 1e-8

        # waveform encoder
        self.encoder = Encoder(win_len, enc_dim)  # [B 2 T]-->[B N 2 L]
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)  # [B N 2 L]-->[B N 2 L]
        self.separator = BF_module(
            self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.layer,
            self.segment_size,
        )
        # [B N 2 L] -> [B E 2 L]
        self.mask_conv1x1 = nn.Conv2d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input: torch.Tensor):
        """
        input: B x 2 x T
        """
        B, num_channel, _ = input.size()
        mixture_w = self.encoder(input)  # [B N 2 L]

        score = self.enc_LN(mixture_w)  # [B N 2 L]
        score = self.separator(score)  # [B E 2 L]
        score = self.mask_conv1x1(score)  # [B E 2 L] -> [B N 2 L]

        est_mask = F.relu(score)

        est_source = self.decoder(mixture_w, est_mask)  # [B E L] + [B E L]--> [B T]

        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]

        return est_source
