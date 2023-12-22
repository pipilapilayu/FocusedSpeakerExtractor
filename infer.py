from dataclasses import dataclass
from typing import Type, TypeVar, Callable
import torch
from DPTNet.models import DPTNet_base
import soundfile
import os
import glob
import tqdm
from dataset import read_wav_44100
from dptnet_modules import DPTNetModule, DPTNetModuleArgs
import lightning
import sys
import pathlib

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


LightningModuleType = TypeVar("LightningModuleType", bound=lightning.LightningModule)


class InferenceServer:
    # we need something to store the state
    def __init__(
        self, module_class: Type[LightningModuleType], model_path: str
    ) -> None:
        self.model = module_class.load_from_checkpoint(model_path)
        self.model.eval()

    @staticmethod
    def process_in_block(
        block_size: int,
        wav: torch.Tensor,
        action: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        wav_len = wav.shape[-1]
        res = torch.zeros_like(wav)
        for i in tqdm.tqdm(range(0, wav_len, block_size)):
            source = wav[..., i : i + block_size]
            estimated_source = action(source)
            res[..., i : i + block_size] += estimated_source
        return res

    def infer(self, wav: Tensor) -> Tensor:
        """
        Args:
            wav: [T] Tensor (not [1 T])
        """

        def action(wav: Tensor) -> Tensor:
            with torch.no_grad():
                return self.model(wav)

        return self.process_in_block(2048 * 300, wav, action)


if __name__ == "__main__":
    server = InferenceServer(DPTNetModule, sys.argv[1])
    for full_filepath in tqdm.tqdm(
        glob.glob("./datasets/dirty/pi/stardew_valley/*.wav")
    ):
        filename = os.path.basename(full_filepath)
        y = read_wav_44100(full_filepath)[..., :5120000]
        y = y.to(device)

        res = server.infer(y)

        # with open("./output/epoch_2/%s.wav" % filename, "wb") as f:
        #     soundfile.write(f, res.squeeze().cpu().numpy(), samplerate=44100, format="WAV")

        res_stereo = torch.cat([y, res], dim=0)
        out_folder = "./output/%s/" % sys.argv[1].replace("/", "_")
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_folder, "stereo_%s.wav" % filename), "wb") as f:
            soundfile.write(
                f,
                res_stereo.cpu().transpose(1, 0).numpy(),
                samplerate=44100,
                format="WAV",
            )
