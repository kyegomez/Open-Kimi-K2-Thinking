import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import RMSNorm


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    base: float = 10000.0,
    original_seq_len: Optional[int] = None,
    rope_factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        dim: Dimensionality of rotary embeddings (must be even).
        seq_len: Maximum sequence length.
        base: Base value for frequency computation.
        original_seq_len: Original sequence length for extended context scaling.
        rope_factor: Scaling factor for extended sequence lengths.
        beta_fast: Fast beta correction factor.
        beta_slow: Slow beta correction factor.

    Returns:
        Precomputed complex exponential values of shape (seq_len, dim // 2).
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot, high_rot, dim, base, max_seq_len
    ):
        low = math.floor(
            find_correction_dim(low_rot, dim, base, max_seq_len)
        )
        high = math.ceil(
            find_correction_dim(high_rot, dim, base, max_seq_len)
        )
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim_size):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (
            torch.arange(dim_size, dtype=torch.float32) - min_val
        ) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    # Apply YaRN scaling if sequence length is extended
    if original_seq_len is not None and seq_len > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim) where head_dim is even.
        freqs_cis: Precomputed complex exponential values of shape (seq_len, head_dim // 2).

    Returns:
        Tensor with rotary embeddings applied, same shape as input.
    """
    dtype = x.dtype
    # Reshape to complex numbers: (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, n_heads, head_dim // 2) as complex
    x_complex = torch.view_as_complex(
        x.float().view(*x.shape[:-1], -1, 2)
    )
    # Expand freqs_cis to match batch and head dimensions: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    seq_len = freqs_cis.size(0)
    head_dim_half = freqs_cis.size(1)
    freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim_half)
    # Apply rotation
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    This is a simplified implementation using pure PyTorch with:
    - LoRA (Low-Rank Adaptation) for query and key-value projections
    - Rotary Positional Embeddings (RoPE)
    - Efficient attention computation using latent key-value caching
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_seq_len: int = 4096,
        max_batch_size: int = 8,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        original_seq_len: Optional[int] = None,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
    ):
        """
        Initialize MLA layer.

        Args:
            dim: Model dimension.
            n_heads: Number of attention heads.
            q_lora_rank: LoRA rank for query projection (0 means no LoRA).
            kv_lora_rank: LoRA rank for key-value projection.
            qk_nope_head_dim: Dimension for query-key without positional embeddings.
            qk_rope_head_dim: Dimension for query-key with rotary embeddings.
            v_head_dim: Dimension for value projections.
            max_seq_len: Maximum sequence length for caching.
            max_batch_size: Maximum batch size for caching.
            rope_theta: Base for rotary positional encoding.
            rope_factor: Scaling factor for extended sequence lengths.
            original_seq_len: Original sequence length for YaRN scaling.
            beta_fast: Fast beta correction factor.
            beta_slow: Slow beta correction factor.
            mscale: Scaling factor for extended attention.
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Query projection (with optional LoRA)
        if q_lora_rank == 0:
            self.wq = nn.Linear(
                dim, n_heads * self.qk_head_dim, bias=False
            )
        else:
            self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_norm = RMSNorm(q_lora_rank)
            self.wq_b = nn.Linear(
                q_lora_rank, n_heads * self.qk_head_dim, bias=False
            )

        # Key-value projection (with LoRA)
        self.wkv_a = nn.Linear(
            dim, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank,
            n_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(n_heads * v_head_dim, dim, bias=False)

        # Attention scaling
        self.softmax_scale = self.qk_head_dim**-0.5
        if (
            original_seq_len is not None
            and max_seq_len > original_seq_len
        ):
            mscale_factor = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale = (
                self.softmax_scale * mscale_factor * mscale_factor
            )

        # Precompute rotary embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                qk_rope_head_dim,
                max_seq_len,
                rope_theta,
                original_seq_len,
                rope_factor,
                beta_fast,
                beta_slow,
            ),
            persistent=False,
        )

        # KV cache for efficient attention
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, max_seq_len, kv_lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(
                max_batch_size, max_seq_len, qk_rope_head_dim
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for MLA.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            start_pos: Starting position in the sequence for caching.
            mask: Optional attention mask of shape (seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)

        # Split query into non-positional and positional parts
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Apply rotary embeddings to positional query
        freqs_cis = self.freqs_cis[start_pos:end_pos]
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Apply rotary embeddings to positional key
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # Update caches
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        # Compute attention scores using cached key-values
        wkv_b = self.wkv_b.weight.view(
            self.n_heads, -1, self.kv_lora_rank
        )
        q_nope_proj = torch.einsum(
            "bshd,hdc->bshc",
            q_nope,
            wkv_b[:, : self.qk_nope_head_dim],
        )

        scores = (
            torch.einsum(
                "bshc,btc->bsht",
                q_nope_proj,
                self.kv_cache[:bsz, :end_pos],
            )
            + torch.einsum(
                "bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos]
            )
        ) * self.softmax_scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask.unsqueeze(1)

        # Softmax
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(
            x
        )

        # Compute output
        x = torch.einsum(
            "bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos]
        )
        x = torch.einsum(
            "bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :]
        )

        # Output projection
        x = self.wo(x.flatten(2))
        return x


# if __name__ == "__main__":
#     # Example usage
#     torch.manual_seed(42)

#     # Model parameters
#     dim = 2048
#     n_heads = 16
#     batch_size = 2
#     seq_len = 128

#     # Create MLA layer
#     mla = MLA(
#         dim=dim,
#         n_heads=n_heads,
#         q_lora_rank=0,  # No LoRA for query
#         kv_lora_rank=512,
#         qk_nope_head_dim=128,
#         qk_rope_head_dim=64,
#         v_head_dim=128,
#         max_seq_len=4096,
#         max_batch_size=batch_size
#     )

#     # Create dummy input
#     x = torch.randn(batch_size, seq_len, dim)

#     # Forward pass
#     output = mla(x, start_pos=0)

#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print("MLA layer created successfully!")
