from open_kimi.model import KimiK2
import torch

if __name__ == "__main__":
    model = KimiK2(
        dim=7168,
        depth=61,
        attention_heads=64,
        experts=384,
        experts_per_token=8,
        seq_len=1024,
        lite_verison=False,
        vocab_size=160000,
    )

    x = torch.randint(0, 10000, (2, 7168))
    out = model(x)
    print(out)
