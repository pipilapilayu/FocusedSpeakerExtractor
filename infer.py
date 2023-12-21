from dataclasses import dataclass
from typing import Type, TypeVar
from DPTNet.solver import Solver
import torch
from DPTNet.models import DPTNet_base
import soundfile
import os
import glob
import tqdm
from dataset import read_wav_44100
from dptnet_modules import DPTNetModule, DPTNetModuleArgs
import lightning

Tensor = torch.Tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class InferArgs:
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


class InferenceServer:
    T = TypeVar("T", bound=lightning.LightningModule)

    # we need something to store the state
    def __init__(self, module_class: Type[T], model_path: str) -> None:
        self.model = module_class.load_from_checkpoint(model_path)
        self.model.eval()

    def infer(self, wav: Tensor) -> Tensor:
        """
        Args:
            wav: [T] Tensor (not [1 T])
        """

        def action(wav: Tensor) -> Tensor:
            with torch.no_grad():
                return self.model(wav)

        return Solver.process_in_block(14 * 44100, wav, action)


if __name__ == "__main__":
    server = InferenceServer(DPTNetModule, "./exp/test/epoch2.pth.tar")
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
