import torch
from open_kimi.kimi_linear import KimiLinear

if __name__ == "__main__":
    model = KimiLinear(
        dim=512,
        num_heads=8,
        head_dim=64,
        chunk_size=64,
        n_experts=16,
        n_activated=4,
        kda_layers=2,
        depth=2,
        vocab_size=10000,
        seq_len=1024,
    )

    x = torch.randint(0, 10000, (2, 1024))

    out = model(x)

    print(out)
    print(out.shape)
