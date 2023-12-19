# Created on 2018/12
# Author: Kaituo XU

import os
from typing import Callable
from loguru import logger
import time
import torch
import numpy
from .others.pit_criterion import cal_loss
from .utils import device
import tqdm


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


class Solver(object):
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.max_norm = args.max_norm
        self.batch_size = args.batch_size
        self.mode = args.mode
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.start_epoch = args.start_epoch
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        self.logger = logger
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print("Loading checkpoint model %s" % self.continue_from)
            self.model.load_state_dict(
                torch.load(self.continue_from, map_location="cpu")
            )
            if "train" in self.mode:
                if self.start_epoch == 0:
                    self.logger.warning(
                        "Loaded a checkpoint file but starting at "
                        "epoch 0. This could lead to an incorrect lr schedule."
                    )
                if self.optimizer.warmup:
                    self.logger.warning(
                        "Loaded a checkpoint file but warming up "
                        "optimizer. This could lead to an incorrect lr schedule."
                    )
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def run(self):
        if "train" not in self.mode:  # just do 1 iteration if not training
            self.epochs = self.start_epoch + 1
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            # Train one epoch
            if "train" in self.mode:
                self.logger.info("Training...")
                self.model.train()  # Turn on BatchNorm & Dropout
                tr_avg_loss = self._run_one_train_epoch(epoch)
                print("-" * 85)
                self.logger.info(
                    "Train Summary | End of Epoch {0} | Time {1:.2f}s | "
                    "Train Loss {2:.3f}".format(
                        epoch + 1, time.time() - start, tr_avg_loss
                    )
                )
                print("-" * 85)
                self.tr_loss[epoch] = tr_avg_loss

                # Save model each epoch
                if self.checkpoint:
                    file_path = os.path.join(
                        self.save_folder, "epoch%d.pth.tar" % (epoch + 1)
                    )
                    torch.save(self.model.state_dict(), file_path)
                    self.logger.info("Saving checkpoint model to %s" % file_path)

                # Learning rate is adjusted in optimizer
                optim_state = self.optimizer.state_dict()
                self.logger.info(
                    "Learning rate at end of epoch {} is {:.6f}".format(
                        epoch, optim_state["param_groups"][0]["lr"]
                    )
                )

            # Cross validation
            if "eval" in self.mode:
                self.logger.info("Cross validation...")
                self.model.eval()  # Turn off Batchnorm & Dropout
                val_loss = self._run_one_eval_epoch(epoch)
                print("-" * 85)
                self.logger.info(
                    "Valid Summary | End of Epoch {0} | Time {1:.2f}s | "
                    "Valid Loss {2:.3f}".format(
                        epoch + 1, time.time() - start, val_loss
                    )
                )
                print("-" * 85)
                self.cv_loss[epoch] = val_loss

            best_file_path = os.path.join(self.save_folder, "temp_best.pth.tar")
            # Save the best model
            if "eval" in self.mode:
                if self.cv_loss[epoch] < self.best_val_loss:
                    self.best_val_loss = self.cv_loss[epoch]
                    torch.save(self.model.state_dict(), best_file_path)
                    self.logger.info(
                        "Found better validated model, saving to %s" % best_file_path
                    )
            elif "train" in self.mode:
                if self.tr_loss[epoch] <= self.tr_loss.min():
                    torch.save(self.model.state_dict(), best_file_path)
                    self.logger.info(
                        "Found better model by train loss, saving to %s"
                        % best_file_path
                    )

    def _run_one_train_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader
        self.logger.info("tr_loader.len() = {}".format(len(data_loader)))
        for i, (data) in enumerate(data_loader):
            padded_mixture_, mixture_lengths_, padded_source_ = data
            seg_idx = numpy.random.randint(0, padded_mixture_.shape[0], self.batch_size)
            padded_mixture = padded_mixture_[seg_idx, :]
            mixture_lengths = mixture_lengths_[seg_idx]
            padded_source = padded_source_[seg_idx, :]
            if self.use_cuda:
                padded_mixture = padded_mixture.to(device)
                mixture_lengths = mixture_lengths.to(device)
                padded_source = padded_source.to(device)
            estimate_source = self.model(padded_mixture)
            si_snr_loss = cal_loss(padded_source, estimate_source, mixture_lengths)
            rms_loss = loudness_loss(estimate_source, padded_source)
            loss = si_snr_loss + rms_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step(epoch)

            total_loss += loss.item()

            if i % self.print_freq == 0:
                logger.info(
                    "Epoch {0} | Iter {1} | "
                    "Current Loss {2:.3f} | SI-SNR {3:.3f} | RMS {4:.3f} | {5:.1f} ms/batch".format(
                        epoch + 1,
                        i + 1,
                        loss.item(),
                        si_snr_loss.item(),
                        rms_loss.item(),
                        1000 * (time.time() - start) / (i + 1),
                    ),
                )

        return total_loss / len(data_loader)

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

    def _run_one_eval_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        data_loader = self.cv_loader
        self.logger.info("cv_loader.len() = {}".format(len(data_loader)))
        for i, (source, source_length) in enumerate(data_loader):
            # we must manually build overlap and manually add back
            if self.use_cuda:
                source = source.to(device)
                source_length = source_length.to(device)

            def action(wav: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    estimate_wav = self.model(wav)
                    torch.cuda.empty_cache()
                return estimate_wav

            estimate_source = self.process_in_block(14 * 44100, source, action)
            si_snr_loss = cal_loss(source, estimate_source, source_length)
            rms_loss = loudness_loss(estimate_source, source)
            loss = si_snr_loss + rms_loss

            total_loss += loss.item()

            if i % self.print_freq == 0:
                logger.info(
                    "Epoch {0} | Iter {1} | "
                    "Current Loss {2:.3f} | SI-SNR {3:.3f} | RMS {4:.3f} | {5:.1f} ms/batch".format(
                        epoch + 1,
                        i + 1,
                        loss.item(),
                        si_snr_loss.item(),
                        rms_loss.item(),
                        1000 * (time.time() - start) / (i + 1),
                    ),
                )

        return total_loss / len(data_loader)
