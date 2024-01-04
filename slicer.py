from bisect import bisect_right
from typing import List, Tuple, Callable, Iterable, Optional
from scipy import signal
import torchaudio
import os
import numpy as np
import pyloudnorm
import soundfile
import tqdm
import click
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from functools import partial


def moving_average(a: np.ndarray, n: int = 3):
    # https://stackoverflow.com/a/14314054
    ret = np.cumsum(a**2, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[(n >> 1) - 1 :] / n) ** 0.5


def normalize_loudness(y: np.ndarray, fs: int, target_loudness=-30) -> np.ndarray:
    meter = pyloudnorm.Meter(fs)
    loudness = meter.integrated_loudness(y)
    normalized = pyloudnorm.normalize.loudness(y, loudness, target_loudness)
    return normalized


def to_mono(y: np.ndarray) -> np.ndarray:
    return np.mean(y, axis=0) if len(y.shape) > 1 else y


def preprocess(y: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
    y_n = np.pad(to_mono(y), fs)
    y_n /= np.max(np.abs(y_n))
    y_n *= 0.9
    return y_n, fs


def load_n_preprocess(filename: str) -> Tuple[np.ndarray, int]:
    y, fs = torchaudio.load(filename)
    return preprocess(y.numpy(), fs)


def sos_filtfilt_highpass(y: np.ndarray, fs: int, fc: int) -> np.ndarray:
    sos = signal.butter(10, fc, "highpass", fs=fs, output="sos")
    y_f = signal.sosfiltfilt(sos, y)
    return y_f


def get_filename(full_filepath: str) -> str:
    return os.path.splitext(os.path.split(full_filepath)[-1])[0]


def clip_to_silence(c: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    # input: list of raw clip ranges, and length of the target audio
    # return: a list with length len(c) + 1
    last_r = 0
    s = []
    for l, r in c:
        s.append((last_r, l))
        last_r = r
    s.append((last_r, n))
    return s


def silence_to_clip(s: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    def l(x):
        return x[0]

    def r(x):
        return x[1]

    return [(r(s[i]), l(s[i + 1])) for i in range(len(s) - 1)]


def update_dp(
    dp: List[int], prev: List[int], s: List[Tuple[int, int]], val: int, i: int, j: int
):
    if dp[i] < val:
        dp[i] = val
        prev[i] = j


def calc_dp(
    s: List[Tuple[int, int]], presum_c: List[int], lower: int, upper: int
) -> Tuple[List[int], List[int]]:
    # input: a list of silence region
    # output: array prev, recording transition between states
    # dp[i] = max clips covered in the merged clip list for clip [0, i), with silence i being selected
    # dp[i] = max(max(dp[j] + num of clips between s[i] and s[j])
    #             if lower <= left boundary of silence i - right boundary of silence j < upper,
    #             max(dp[j]) for all s[j] with boundaries that failed to meet the constraints
    def l(x):
        return x[0]

    def r(x):
        return x[1]

    dp = [0] * len(s)
    prev = [-1] * len(s)
    for i in range(1, len(s)):
        max_j = bisect_right(s, l(s[i]) - lower, hi=i, key=r)
        update_dp(dp, prev, s, dp[i - 1], i, i - 1)
        for j in range(max_j - 1, -1, -1):
            if not lower <= l(s[i]) - r(s[j]) < upper:
                break
            # update_dp(dp, prev, s, dp[j] + i - j, i, j)
            update_dp(dp, prev, s, dp[j] + presum_c[i] - presum_c[j], i, j)
    return dp, prev


def split_by_prev(
    s: List[Tuple[int, int]], prev: List[int], n: int, lower: int, upper: int
) -> List[Tuple[int, int]]:
    last_l = n
    i = len(s) - 1
    optimal_split: List[Tuple[int, int]] = []
    while i >= 0:
        l, r = s[i]
        if lower <= last_l - r < upper:
            optimal_split.append((r, last_l))
        last_l = l
        i = prev[i]
    optimal_split.reverse()
    return optimal_split


def presum(a: Iterable[int]) -> List[int]:
    # sum(a[l:r]) = presum_a[r] - presum_a[l]
    # thus, presum_a[i] = sum(a[:i])
    c = [0]
    for v in a:
        c.append(c[-1] + v)
    return c


def merge_by_dp(
    c: List[Tuple[int, int]], fs: int, n: int, lower: int = 5, upper: int = 15
) -> List[Tuple[int, int]]:
    # c: raw clips, fs: sampling freqency, n: length of input signal
    s = clip_to_silence(c, n)
    dp, prev = calc_dp(
        s, presum(map(lambda x: (lambda l, r: r - l)(*x), c)), lower * fs, upper * fs
    )
    split_scheme = split_by_prev(s, prev, n, lower * fs, upper * fs)
    return split_scheme


def merge_by_dynamic_gap(
    c: List[Tuple[int, int]], fs: int, n: int, lower: int = 5, upper: int = 15
) -> List[Tuple[int, int]]:
    # dynamic gap mechanism ensures average RMS to be always larger than threshold
    def dynamic_gap(last_range: Tuple[int, ...], fs: int):
        return min(7.5 * fs - 0.5 * (last_range[1] - last_range[0]), 5 * fs)

    def merge_small_gaps(
        stack: List[Tuple[int, int, int]],
        cur_range: Tuple[int, int],
        lower=0.5,
    ) -> None:
        # threshold: max interval between last offset and this on set
        last_range = stack[-1]
        if cur_range[1] - last_range[0] < upper * fs and cur_range[0] - last_range[
            1
        ] < dynamic_gap(last_range, fs):
            stack.pop()
            stack.append(
                (
                    last_range[0],
                    cur_range[1],
                    last_range[2] + cur_range[0] - last_range[1],
                )
            )
        else:
            if (last_range[1] - last_range[0]) < lower * fs:
                stack.pop()
            stack.append((*cur_range, 0))

    stack: List[Tuple[int, int, int]] = []
    for l, r in c:
        if stack:
            merge_small_gaps(stack, (l, r), lower=lower)
        else:
            stack.append((l, r, 0))

    filtered_stack = list(
        filter(lambda x: (lambda l, r, _: l < r and (r - l) / fs >= lower)(*x), stack)
    )
    return list(map(lambda x: x[:2], filtered_stack))


def no_merge(
    c: List[Tuple[int, int]], fs: int, n: int, lower: int = 5, upper: int = 15
):
    filtered_stack = list(
        filter(lambda x: (lambda l, r: l < r and lower <= (r - l) / fs < upper)(*x), c)
    )
    return filtered_stack


def get_clips(
    y: np.ndarray, fs: int, threshold: float = 0.002
) -> List[Tuple[int, int]]:
    y_avg = moving_average(y, int(fs * 0.3))
    mask = y_avg >= threshold

    mask_diff = np.diff(mask.astype(int))
    onset = np.where(mask_diff == 1)[0]
    offset = np.where(mask_diff == -1)[0]

    if len(onset) == len(offset) + 1:
        offset = np.append(offset, len(y))

    c = list(zip(onset, offset))
    return c


def split_file(
    y: np.ndarray,
    fs: int,
    merger: Callable[[List[Tuple[int, int]], int, int], List[Tuple[int, int]]],
    fc: Optional[int] = None,
) -> List[Tuple[np.ndarray, int, int]]:
    return list(
        (
            normalize_loudness(
                sos_filtfilt_highpass(y[l:r], fs, fc) if fc is not None else y[l:r],
                fs,
                -23,
            ),
            l,
            r,
        )
        for l, r in tqdm.tqdm(merger(get_clips(y, fs), fs, len(y)))
    )


def remove_silence(
    y: np.ndarray, c: List[Tuple[int, int]], margin: int = 8820
) -> np.ndarray:
    short_s = list(
        filter(
            lambda x: (lambda l, r: r > l)(*x),
            map(
                lambda x: (lambda l, r: (l + margin, r - margin))(*x),
                clip_to_silence(c, len(y)),
            ),
        )
    )
    res_c = silence_to_clip(short_s)
    if not res_c:
        res_c.append((0, len(y)))
    return np.concatenate([y[l:r] for l, r in res_c])


def split_single_file_no_caption(
    full_filepath: str,
    output_folder: str,
    should_remove_silence: bool = False,
    low_cut_freq: Optional[int] = None,
):
    filename = get_filename(full_filepath)
    y, fs = load_n_preprocess(full_filepath)
    if should_remove_silence:
        y = remove_silence(y, get_clips(y, fs))

    for i, (y_n, l, r) in enumerate(split_file(y, fs, merge_by_dp, low_cut_freq)):
        with open("%s/%s_%d.wav" % (output_folder, filename, i), "wb") as f:
            soundfile.write(f, y_n, fs, format="WAV")


def split_single_file_with_caption(
    generate_caption_for: str,
    full_filepath: str,
    output_folder: str,
    should_remove_silence: bool = False,
    low_cut_freq: Optional[int] = None,
):
    filename = get_filename(full_filepath)
    y, fs = load_n_preprocess(full_filepath)
    if should_remove_silence:
        y = remove_silence(y, get_clips(y, fs))

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )

    with open(
        os.path.join(output_folder, "%s.list" % generate_caption_for),
        "a",
        encoding="utf-8",
    ) as f:
        for i, (y_n, l, r) in enumerate(split_file(y, fs, merge_by_dp, low_cut_freq)):
            out_filename = "%s_%d.wav" % (filename, i)
            with open(os.path.join(output_folder, out_filename), "wb") as g:
                infer_res = inference_pipeline(audio_in=y_n, audio_fs=fs)
                if "text" in infer_res:
                    f.write(
                        "|".join(
                            [
                                os.path.join(
                                    os.path.abspath(output_folder), out_filename
                                ),
                                generate_caption_for,
                                "ZH",
                                infer_res["text"],
                            ]
                        )
                        + "\n"
                    )
                    soundfile.write(g, y_n, fs, format="WAV")


def split_single_file(
    full_filepath: str,
    output_folder: str,
    should_remove_silence: bool = False,
    generate_caption_for: Optional[str] = None,
    low_cut_freq: Optional[int] = None,
):
    (
        partial(split_single_file_with_caption, generate_caption_for)
        if generate_caption_for is not None
        else split_single_file_no_caption
    )(full_filepath, output_folder, should_remove_silence, low_cut_freq)


@click.command()
@click.argument(
    "src", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument(
    "dst", nargs=1, type=click.Path(exists=False, file_okay=False, dir_okay=True)
)
@click.option(
    "--remove_silence", type=bool, default=False, help="Remove silence or not"
)
@click.option(
    "--generate_caption_for",
    type=str,
    default=None,
    help="Generate filelist compatible with BERT-VITS-2 with specified speaker name",
)
def main(
    src: Iterable[str],
    dst: str,
    remove_silence: bool,
    generate_caption_for: Optional[str],
):
    for fn in src:
        split_single_file(fn, dst, remove_silence, generate_caption_for)


if __name__ == "__main__":
    main()
