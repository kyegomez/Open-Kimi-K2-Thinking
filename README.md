# Open-Kimi-K2-Thinking

![Architecture kimi k2](image/arch.png)

This repository is a straightforward attempt to implement the base Kimi K2 Reasoning model architecture in pure PyTorch as simply as possible.

[Link](https://huggingface.co/moonshotai/Kimi-K2-Thinking)


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

## Post Training

On the model huggingface page, they mention they use Native INT4 Quantization in the post training phase. So I would say a good post training recipe would include:

- Native INT4 Quantization
- MUON Optimizer
- GRPO


# Citation

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
