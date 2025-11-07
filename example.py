from open_kimi.model import KimiK2
import torch

if __name__ == "__main__":
    model = KimiK2(
        dim=512,
        depth=2,
        attention_heads=8,
        experts=16,
        experts_per_token=4,
        seq_len=1024,
        lite_verison=True,
        vocab_size=10000,
    )

    x = torch.randint(0, 10000, (2, 1024))
    out = model(x)
    print(out)
