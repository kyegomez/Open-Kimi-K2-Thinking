"""Kimi K2 model implementation.

This module contains the core components of the Kimi K2 language model,
including the TransformerBlock and the full KimiK2 model architecture.
"""

from typing import Any, Optional

import torch
from torch import nn

from open_kimi.mla import MLA
from open_kimi.moe import MoE
from torch import Tensor


class TransformerBlock(nn.Module):
    """A single transformer block for the Kimi K2 model.

    This block consists of:
    - Multi-head Linear Attention (MLA) mechanism
    - Mixture of Experts (MoE) feed-forward network
    - RMSNorm for layer normalization
    - Residual connections

    The block supports a "lite" version that reduces the number of experts
    and sequence length for more efficient computation.
    """

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
        """Initialize a TransformerBlock.

        Args:
            dim: The dimension of the model embeddings.
            attention_heads: Number of attention heads for the MLA layer.
                Defaults to 16.
            experts: Number of experts in the MoE layer. Defaults to 384.
            experts_per_token: Number of experts to activate per token.
                Defaults to 8.
            seq_len: Maximum sequence length for the attention mechanism.
                Defaults to 256052.
            lite_verison: If True, uses reduced parameters (4 experts,
                2 experts per token, seq_len 1024) for efficiency.
                Defaults to True.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).
        """
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

        self.norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Any]) -> Tensor:
        """Forward pass through the transformer block.

        The forward pass follows this structure:
        1. Apply RMSNorm and then multi-head linear attention
        2. Add residual connection
        3. Apply RMSNorm and then MoE feed-forward
        4. Add residual connection

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            mask: Optional attention mask. Currently unused but kept for
                API compatibility.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim) after
            applying attention and MoE transformations with residual connections.
        """
        original = x

        attended = self.attn((self.norm(x)))

        second_layer = original + attended

        mixed = self.moe(self.norm(second_layer))

        return second_layer + mixed


class KimiK2(nn.Module):
    """The full Kimi K2 language model architecture.

    This model consists of:
    - Token embeddings
    - A stack of TransformerBlocks
    - An output head that projects to vocabulary logits

    The model supports both full and lite configurations, with the lite
    version using reduced parameters for more efficient computation.
    """

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
        post_embed_norm: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the KimiK2 model.

        Args:
            dim: The dimension of the model embeddings.
            depth: Number of transformer blocks to stack. Defaults to 61.
            attention_heads: Number of attention heads per block.
                Defaults to 64.
            experts: Number of experts in each MoE layer. Defaults to 384.
            experts_per_token: Number of experts to activate per token.
                Defaults to 8.
            seq_len: Maximum sequence length for the attention mechanism.
                Defaults to 256052.
            lite_verison: If True, uses reduced parameters in each block
                for efficiency. Defaults to True.
            vocab_size: Size of the vocabulary for token embeddings and
                output projection. Defaults to 160000.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.attention_heads = attention_heads
        self.experts = experts
        self.experts_per_token = experts_per_token
        self.seq_len = seq_len
        self.lite_verison = lite_verison
        self.post_embed_norm = post_embed_norm

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )
        
        self.norm = nn.RMSNorm(dim)

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
            nn.RMSNorm(dim),
            nn.Linear(dim, vocab_size),
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        """Forward pass through the KimiK2 model.

        The forward pass:
        1. Embeds input token indices
        2. Creates a causal attention mask if sequence length > 1
        3. Passes through all transformer blocks
        4. Applies output head to get vocabulary logits

        Args:
            x: Input tensor of token indices with shape
                (batch_size, seq_len).

        Returns:
            Output tensor of logits with shape
            (batch_size, seq_len, vocab_size).
        """
        seqlen = x.size(1)

        x = self.embedding(x)
        
        if self.post_embed_norm:
            x = self.norm(x)

        mask = None

        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=x.device
            ).triu_(1)

        for block in self.blocks:
            x = block(x, mask=mask)

        return self.output_head(x)
