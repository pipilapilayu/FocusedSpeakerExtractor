from dataclasses import dataclass
from typing import Callable
from DPTNet.solver import Solver
import torch
import torchaudio
from main import MainArgs
from DPTNet.models import DPTNet_base
import soundfile
import os
import glob
import tqdm
from dataset import read_wav_44100

Tensor = torch.Tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class DPTNetArgs:
    n: int = 64  # feature dim in DPT blocks
    w: int = 2  # filter length in encoder
    k: int = 250  # chunk size in frames
    d: int = 6  # number of DPT blocks
    h: int = 4  # number of hidden units in LSTM after multihead attention
    e: int = 256  # #channels before bottleneck


class InferenceServer:
    # we need something to store the state
    def __init__(self, model: DPTNet_base, model_path: str) -> None:
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def infer(self, wav: Tensor) -> Tensor:
        """
        Args:
            wav: [T] Tensor (not [1 T])
        """

        def action(wav: Tensor) -> Tensor:
            with torch.no_grad():
                return model(wav)

        return Solver.process_in_block(14 * 44100, wav, action)


if __name__ == "__main__":
    args = DPTNetArgs(w=16, d=2)
    model = DPTNet_base(
        enc_dim=args.e,
        feature_dim=args.n,
        hidden_dim=args.h,
        layer=args.d,
        segment_size=args.k,
        win_len=args.w,
    )
    model.cuda()
    model.to(device)
    server = InferenceServer(model, "./exp/test/epoch2.pth.tar")
    for full_filepath in tqdm.tqdm(
        glob.glob("./datasets/dirty/pi/stardew_valley/*.m4a")
    ):
        filename = os.path.basename(full_filepath)
        y = read_wav_44100(full_filepath)[..., :5000000]
        y = y.to(device)

        res = server.infer(y)

        # with open("./output/epoch_2/%s.wav" % filename, "wb") as f:
        #     soundfile.write(f, res.squeeze().cpu().numpy(), samplerate=44100, format="WAV")

        res_stereo = torch.cat([y, res], dim=0)
        with open("./output/epoch_2/stereo_%s.wav" % filename, "wb") as f:
            soundfile.write(
                f,
                res_stereo.cpu().transpose(1, 0).numpy(),
                samplerate=44100,
                format="WAV",
            )
