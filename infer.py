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

from settings import FS

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


if __name__ == "__main__":
    server = InferenceServer(N2NDPTNetModule, sys.argv[1])
    for full_filepath in tqdm.tqdm(
        glob.glob("./datasets/dirty/pi/stardew_valley/*.wav")
    ):
        filename = os.path.basename(full_filepath)
        y = read_wav_at_FS(full_filepath)[..., :9600000].to(device).unsqueeze(dim=0)

        res = server.infer(y)

        out_folder = "./output/%s/" % sys.argv[1].replace("/", "_")
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_folder, "%s.wav" % filename), "wb") as f:
            soundfile.write(
                f,
                res.squeeze().transpose(1, 0).cpu().numpy(),
                samplerate=FS,
                format="WAV",
            )

        # res_stereo = torch.cat([y, res], dim=0)
        # out_folder = "./output/%s/" % sys.argv[1].replace("/", "_")
        # pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        # with open(os.path.join(out_folder, "stereo_%s.wav" % filename), "wb") as f:
        #     soundfile.write(
        #         f,
        #         res_stereo.transpose(1, 0).cpu().numpy(),
        #         samplerate=FS,
        #         format="WAV",
        #     )
