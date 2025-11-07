from typing import Any, Optional

import torch
from torch import nn

from open_kimi.mla import MLA
from open_kimi.moe import MoE
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        attention_heads: int = 16,
        experts: int = 384,
        experts_per_token: int = 8,
        seq_len: int = 256052,
        lite_verison: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.attention_heads = attention_heads
        self.experts = experts
        self.experts_per_token = experts_per_token
        self.seq_len = seq_len
        self.lite_verison = lite_verison

        if self.lite_verison:
            experts = 4
            experts_per_token = 2
            seq_len = 1024

        self.attn = MLA(
            dim=dim,
            n_heads=attention_heads,
            max_seq_len=seq_len,
        )

        self.moe = MoE(
            dim=dim,
            n_experts=experts,
            n_activated=experts_per_token,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Any]) -> Tensor:
        original = x

        attended = self.attn((self.norm(x)))

        second_layer = original + attended

        mixed = self.moe(self.norm(second_layer))

        return second_layer + mixed


class KimiK2(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 61,
        attention_heads: int = 64,
        experts: int = 384,
        experts_per_token: int = 8,
        seq_len: int = 256052,
        lite_verison: bool = True,
        vocab_size: int = 160000,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.attention_heads = attention_heads
        self.experts = experts
        self.experts_per_token = experts_per_token
        self.seq_len = seq_len
        self.lite_verison = lite_verison

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    attention_heads=attention_heads,
                    experts=experts,
                    experts_per_token=experts_per_token,
                    seq_len=seq_len,
                    lite_verison=lite_verison,
                )
                for _ in range(depth)
            ]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        seqlen = x.size(1)

        x = self.embedding(x)

        mask = None

        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=x.device
            ).triu_(1)

        for block in self.blocks:
            x = block(x, mask=mask)

        return self.output_head(x)
