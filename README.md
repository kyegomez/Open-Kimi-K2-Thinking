# Open-Kimi-K2-Thinking

![Architecture kimi k2](image/arch.png)

This repository is a straightforward attempt to implement the base Kimi K2 Reasoning model architecture in pure PyTorch as simply as possible.

[Link](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

## Install

```bash
pip3 install -U open-kimi
```

## Example

```python
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
```

## Full Example

```python
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
```

## Kimi Linear

![Kimi Linear Architecture](image/linear.png)

Kimi Linear is a hybrid linear attention architecture that outperforms full attention under fair comparisons across various scenarios, including short-context, long-context, and reinforcement learning scaling regimes. At its core is **Kimi Delta Attention (KDA)**, an expressive linear attention module that extends Gated DeltaNet with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. **Paper Link**: [Kimi Linear: An Expressive, Efficient Attention Architecture](https://huggingface.co/papers/2510.26692) (arXiv:2510.26692)

### Usage Example

```python
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
```

## Post Training

On the model huggingface page, they mention they use Native INT4 Quantization in the post training phase. So I would say a good post training recipe would include:

- Native INT4 Quantization
- MUON Optimizer
- GRPO

## Citation

```bibtex
@misc{moonshot-kimi-k2,
  title={Kimi K2 Thinking},
  author={Moonshot AI},
  year={2024},
  howpublished={\url{https://huggingface.co/moonshotai/Kimi-K2-Thinking}}
}
```

## Acknowledgments

This implementation is based on the architecture specifications published by Moonshot AI for the Kimi K2 Thinking model. Special thanks to the Moonshot AI team for making the model architecture details publicly available.

## Contact

For questions, issues, or contributions, please open an issue on the repository or contact the maintainers.

---

**Note**: This is an independent implementation based on publicly available specifications. It is not affiliated with or endorsed by Moonshot AI. For production use, please refer to the official model repository and weights.
