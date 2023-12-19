import torch
from thop import profile
from thop import clever_format
from models import DPTNet_base


def main():
    model = DPTNet_base(
        enc_dim=256,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=250,
        nspk=2,
        win_len=2,
    )

    input = torch.rand(1, 32000)
    output = model(input)
    flops, params = profile(model, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("output shape:", output.shape)
    print("model size:", params)


if __name__ == "__main__":
    main()
