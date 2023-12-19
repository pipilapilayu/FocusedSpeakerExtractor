# FocusedSpeechExtractor

A model that derives from TSE works, and aims to solely extract one speaker's speech from all kinds of noises, including other people's voice, game SFX, noise, etc.

Serve as fundamental model for downstream tasks, such as ASR, SVC, SVS, TTS, etc.

Unlike most TSE research that runs at 8kHz, the model runs at 44.1kHz to ensure the output could be further utilized.

## How to use

The project is still at it's early stage. Currently we are still expanding our dataset. So currently we focus on extracting more clean data to bootstrap.

You need clean data and dirty data to do train this model. Here's an example of the layout:

```
datasets
├── clean
│   └── spk0
│       └── bootstrap
│           ├── src_0_0.wav
│           ├── src_0_1.wav
│           ├── src_0_10.wav
│           ├── src_0_100.wav
│           ├── ...
│           └── src_5_99.wav
└── dirty
    ├── other
    │   └── stardew_valley
    │       ├── src_0.wav
    │       └── src_1.wav
    └── spk0
        └── stardew_valley
            ├── src_0.wav
            ├── src_1.wav
            └── src_2.wav
```

Here we use spk0's clean data (bootstrap) and other's stardew_valley noise to build a extractor that's specifically targeted to extract spk0's speech mixed with stardew valley.

Make sure all files in spk0 has length in 5-15s, otherwise you may get CUDA OOM. You may use `slicer.py` to slice some long, clean audio that only contains target speaker's voice into small segments. You may find more information on the slicer at the end of README.

The overall procedure could be divided into two phase:

### Phase I: Train extractor

```bash
python train.py
```

Unfortunately we still doesn't support command line as we are still prototyping things. You must edit the `MainArgs` object passed into the `main` function in `train.py`.

### Phase II: Infer

```bash
python infer.py
```

If you want to infer some file, you must manually edit the `infer.py` and remove that `glob.glob()` and manually specify the path to the file.

You also need to make sure you use the same model args you used when training.

## Mixture Dataset

We first load a clean wav, then apply offset to shift it left or right meanwhile maintaining it's shape by zero padding at left or right side.

We then randomly pick a segment with the same length as the clean wav, from random file in all provided dirty folders.

Further augmentation is under research.

## Extractor

We use `DPTNet` as the speech extractor for now. We plan to do the following experiments in the future:

- [ ] Apply [sigma-reparam](https://github.com/apple/ml-sigma-reparam.git) to the transformers
- [ ] Substitute transformer block with [RWKV](https://github.com/BlinkDL/RWKV-LM.git)
- [ ] Try frequency-domain solutions (e.g. diffusion-based approach)
- [ ] Separate model-specific args from training args and infer args
- And most importantly, read more papers...

## Bibliography

- [DPTNet](https://arxiv.org/pdf/2007.13975.pdf)
- [Forked DPTNet](https://github.com/ilyakava/DPTNet)
