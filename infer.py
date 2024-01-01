from dataclasses import dataclass
from typing import Type, TypeVar, Callable
import torch
from DPTNet.models import DPTNet_base
import soundfile
import os
import glob
import tqdm
from dataset import read_wav_at_FS
from dptnet_modules import N2NDPTNetModule, DPTNetModuleArgs
from torch.utils.data import TensorDataset
import lightning
import sys
import pathlib

from settings import infer_block_size, FS

Tensor = torch.Tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LightningModuleType = TypeVar("LightningModuleType", bound=lightning.LightningModule)


class InferenceServer:
    # we need something to store the state
    def __init__(
        self, module_class: Type[LightningModuleType], model_path: str
    ) -> None:
        self.model = module_class.load_from_checkpoint(model_path)
        self.model.eval()

    def infer(self, wav: Tensor) -> Tensor:
        """
        Args:
            wav: 1 x T
        """

        result = self.model.predict_step(wav)
        return result


def infer_one_file(full_filepath: str, out_folder: str):
    filename = os.path.basename(full_filepath)
    y = read_wav_at_FS(full_filepath)
    y_len = y.shape[-1]
    pad_y_len = y_len + (infer_block_size - y_len % infer_block_size) % infer_block_size
    padded_y = torch.zeros(*y.shape[:-1], pad_y_len)
    padded_y[..., :y_len] = y

    res = server.infer(padded_y.to(device))

    # with open("./output/epoch_2/%s.wav" % filename, "wb") as f:
    #     soundfile.write(f, res.squeeze().cpu().numpy(), samplerate=44100, format="WAV")

    res_stereo = torch.cat([padded_y, res.cpu()], dim=0)
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_folder, "stereo_%s.wav" % filename), "wb") as f:
        soundfile.write(
            f,
            res_stereo.transpose(1, 0).cpu().numpy(),
            samplerate=FS,
            format="WAV",
        )


if __name__ == "__main__":
    server = InferenceServer(N2NDPTNetModule, sys.argv[1])
    out_folder = "./output/%s/" % sys.argv[1].replace("/", "_")
    for full_filepath in tqdm.tqdm(
        glob.glob("./datasets/dirty/mixed/stardew_valley/*.m4a")
    ):
        infer_one_file(full_filepath, out_folder)
