from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from open_kimi.mla import MLA
from open_kimi.moe import MoE


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunkwise KDA implementation from Listing 1 (Appendix C) of the paper.

    Reference: "Kimi Linear: An Expressive, Efficient Attention Architecture"
    Paper: arXiv:2510.26692v2, Appendix C, Listing 1 (right panel, lines 1-30)

    This is the PyTorch-style pseudocode implementation that achieves ~2× speedup
    over DPLR by binding a=b=k (§6.2, Figure 2).

    Args:
        q: Query tensor [B, H, T, K] - after L2 normalization
        k: Key tensor [B, H, T, K] - after L2 normalization
        v: Value tensor [B, H, T, V] - after Swish activation
        g: Log-space cumulative decay [B, H, T, K] - log(α_t) accumulated
        beta: Learning rate [B, H, T] - for delta rule (not used in simplified version)
        chunk_size: Chunk size C (default 64)

    Returns:
        o: Output tensor [B, H, T, V]
        S: Final state [B, H, K, V]
    """
    # Line 2: Extract dimensions
    B, H, T, K = q.shape
    V = v.shape[-1]
    BT = chunk_size  # C in paper notation

    # Line 3: Initialize number of chunks NT and state S
    NT = T // BT
    S = k.new_zeros(B, H, K, V)  # Recurrent state S ∈ R^{K×V}

    # Line 4: Reshape into chunks [B, H, NT, BT, K]
    # This converts [B, H, T, K] → [B, H, N, C, K] where N=NT, C=BT
    q = rearrange(q, "b h (n c) k -> b h n c k", c=BT)
    k = rearrange(k, "b h (n c) k -> b h n c k", c=BT)
    v = rearrange(v, "b h (n c) v -> b h n c v", c=BT)
    g = rearrange(g, "b h (n c) k -> b h n c k", c=BT)

    # Line 5: Cumulative sum of log-decay within each chunk
    # gc[i] = Σ_{j=1}^i log(α_j), so exp(gc[i]) = γ_i = ∏_{j=1}^i α_j
    gc = g.cumsum(-2)  # [B, H, NT, BT, K]

    # Line 6: Allocate attention score matrices
    # Aqk: Query-Key attention scores [B, H, NT, BT, BT]
    # Akk: Key-Key interaction matrix [B, H, NT, BT, BT]
    Aqk = torch.zeros(
        B, H, NT, BT, BT, device=q.device, dtype=q.dtype
    )
    Akk = torch.zeros(
        B, H, NT, BT, BT, device=q.device, dtype=q.dtype
    )

    # Lines 8-15: Compute Aqk and Akk matrices
    # This computes attention scores with exponential decay
    for i in range(BT):
        # Line 9: Extract i-th query and key (with singleton dimension for broadcasting)
        k_i = k[:, :, :, i, None, :]  # [B, H, NT, 1, K]
        q_i = q[:, :, :, i, None, :]  # [B, H, NT, 1, K]

        # Line 10: Extract cumulative decay at position i
        g_i = gc[:, :, :, i : i + 1, :]  # [B, H, NT, 1, K]

        # Line 11: Causal mask - only attend to positions ≤ i
        mask = (torch.arange(BT, device=q.device) <= i)[
            None, None, None, :, None
        ]  # [1, 1, 1, BT, 1]

        # Line 12: s1_i = exp(g_i - gc) for j ≤ i, else 0
        # This is the decay from position i to position j: γ_j / γ_i
        s1_i = (g_i - gc).exp()  # [B, H, NT, BT, K]
        s1_i = s1_i.where(
            mask, torch.tensor(0.0, device=q.device, dtype=q.dtype)
        )

        # Line 13: s2_i = exp(gc - g_i)
        # This is the decay from position j to position i: γ_i / γ_j
        s2_i = (gc - g_i).exp()  # [B, H, NT, BT, K]

        # Line 14: Aqk[i,j] = q_i^T (γ_j/γ_i) k_j = q_i^T s1_i k
        # Query-key attention scores with decay
        Aqk[:, :, :, i, :] = (q_i * k * s1_i).sum(
            -1
        )  # [B, H, NT, BT]

        # Line 15: Akk[i,j] = k_i^T (γ_i/γ_j) k_j = k_i^T s2_i k
        # Key-key interaction for delta rule correction
        Akk[:, :, :, i, :] = (k_i * k * s2_i).sum(
            -1
        )  # [B, H, NT, BT]

    # Line 16: Upper triangular mask (including diagonal)
    mask = torch.triu(
        torch.ones(BT, BT, device=q.device, dtype=torch.bool),
        diagonal=0,
    )

    # Line 17: A = -Akk with upper triangular part masked to 0
    # This creates the strictly lower triangular matrix for delta rule
    A = -Akk.masked_fill(mask, 0)  # [B, H, NT, BT, BT]

    # Lines 18-19: Forward substitution to compute (I + A)^{-1}
    # This is Gaussian elimination for lower triangular systems
    # A[i, :i] += A[i, :] @ A[:, :i] iteratively
    for i in range(1, BT):
        A[:, :, :, i, :i] = A[:, :, :, i, :i] + (
            A[:, :, :, i, :, None] * A[:, :, :, :, :i].clone()
        ).sum(-2)

    # Line 20: Add identity: A = (I + A)
    # Now A represents M = (I - Akk)^{-1} from the WY representation
    A = A + torch.eye(BT, device=q.device, dtype=q.dtype)

    # Line 21: Compute W and U matrices (Eq. 7)
    # w = A @ (exp(gc) ⊙ k): corrected keys with decay
    # u = A @ v: corrected values
    w = A @ (gc.exp() * k)  # [B, H, NT, BT, K]
    u = A @ v  # [B, H, NT, BT, V]

    # Line 22: Initialize output tensor
    o = torch.zeros_like(v)  # [B, H, NT, BT, V]

    # Line 23: Strictly upper triangular mask (for intra-chunk causal attention)
    mask = torch.triu(
        torch.ones(BT, BT, device=q.device, dtype=torch.bool),
        diagonal=1,
    )

    # Lines 24-29: Process each chunk
    for i in range(NT):
        # Line 25: Extract i-th chunk for each tensor
        q_i = q[:, :, i]  # [B, H, BT, K]
        k_i = k[:, :, i]  # [B, H, BT, K]
        u_i = u[:, :, i]  # [B, H, BT, V]
        g_i = gc[:, :, i]  # [B, H, BT, K]
        w_i = w[:, :, i]  # [B, H, BT, K]

        # Line 26: Compute output for chunk i (Eq. 9)
        # o = (q ⊙ exp(g)) @ S + Aqk @ (u - w @ S)
        # First term: inter-chunk attention (query the recurrent state)
        # Second term: intra-chunk attention with delta rule correction
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk[:, :, i] @ (
            u_i - w_i @ S
        )

        # Line 27: Compute decay for state update
        # decay = exp(g_i[:, :, -1:] - g_i)
        # This is γ_C / γ_j for each position j in the chunk
        decay = (g_i[:, :, -1:] - g_i).exp()  # [B, H, BT, K]

        # Line 28: Apply exponential decay to state
        # S = S * exp(g_i[:, :, -1, :])
        # Multiply by γ_C to decay the entire state
        S = S * g_i[:, :, -1, :, None].exp()  # [B, H, K, V]

        # Line 29: Add contribution from current chunk (Eq. 8)
        # S += (k ⊙ decay)^T @ v
        # This accumulates the key-value pairs with appropriate decay
        S = S + (k_i * decay).transpose(-1, -2) @ u_i  # [B, H, K, V]

    # Line 30: Reshape output back to [B, H, T, V] and return
    o = rearrange(o, "b h n c v -> b h (n c) v")
    return o, S


class KimiDeltaAttention(nn.Module):
    """
    Complete Kimi Delta Attention module matching Figure 3 (right panel).

    This implements the full architecture with:
    - Input processing: Linear → Conv → L2/Swish (bottom 3 branches)
    - Fine-grained decay α_t: Low-rank projection (middle-right trapezoid)
    - Delta rule with β_t: Learning rate for state updates
    - Output gating: Low-rank projection (far-right trapezoid)
    - Head-wise RMSNorm: Post-attention normalization
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 128,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        # === Bottom 3 Branches in Figure 3 ===

        # Branch 1 (left): q, k processing
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.conv_q = nn.Conv1d(
            num_heads * head_dim,
            num_heads * head_dim,
            kernel_size=4,
            padding=3,
            groups=num_heads * head_dim,
        )
        self.conv_k = nn.Conv1d(
            num_heads * head_dim,
            num_heads * head_dim,
            kernel_size=4,
            padding=3,
            groups=num_heads * head_dim,
        )
        # L2 normalization applied in forward pass

        # Branch 2 (middle): v processing
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.conv_v = nn.Conv1d(
            num_heads * head_dim,
            num_heads * head_dim,
            kernel_size=4,
            padding=3,
            groups=num_heads * head_dim,
        )
        # Swish (SiLU) activation applied in forward pass

        # Branch 3a (middle-right trapezoid): α_t decay gate
        # Low-rank: dim → head_dim → (num_heads × head_dim)
        decay_rank = head_dim
        self.decay_proj_down = nn.Linear(dim, decay_rank, bias=False)
        self.decay_proj_up = nn.Linear(
            decay_rank, num_heads * head_dim, bias=False
        )

        # Branch 3b: β_t learning rate (not shown in bottom but feeds into KDA)
        self.beta_proj = nn.Linear(dim, num_heads, bias=False)

        # Branch 4 (far-right trapezoid): Output gate
        # Low-rank: dim → head_dim → (num_heads × head_dim)
        self.gate_proj_down = nn.Linear(dim, decay_rank, bias=False)
        self.gate_proj_up = nn.Linear(
            decay_rank, num_heads * head_dim, bias=False
        )

        # Top of diagram: Norm → Linear → Output
        self.head_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass matching Figure 3 architecture.

        Flow:
        1. Input splits into 4 branches
        2. Branch 1: x → Linear → Conv → L2 → q, k
        3. Branch 2: x → Linear → Conv → Swish → v
        4. Branch 3: x → W↓_α → W↑_α → Sigmoid → α_t
        5. Branch 4: x → W↓_g → W↑_g → Sigmoid → gate
        6. All feed into KDA block
        7. Output: KDA → Norm → gate ⊙ · → Linear → output
        """
        B, T, D = x.shape

        # === Process Branch 1: q, k (Linear → Conv → L2) ===
        q = self.q_proj(x)  # [B, T, H*K]
        k = self.k_proj(x)

        # Short convolution
        q = self.conv_q(q.transpose(1, 2))[:, :, :T].transpose(1, 2)
        k = self.conv_k(k.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Reshape and L2 normalize (Figure 3: "L2" box)
        q = rearrange(q, "b t (h k) -> b h t k", h=self.num_heads)
        k = rearrange(k, "b t (h k) -> b h t k", h=self.num_heads)
        q = F.normalize(q, p=2, dim=-1)  # L2 normalization
        k = F.normalize(k, p=2, dim=-1)

        # === Process Branch 2: v (Linear → Conv → Swish) ===
        v = self.v_proj(x)
        v = self.conv_v(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        v = F.silu(v)  # Swish activation (Figure 3: "σ" box)
        v = rearrange(v, "b t (h k) -> b h t k", h=self.num_heads)

        # === Process Branch 3a: α_t decay (Low-rank → Sigmoid) ===
        # This is the middle-right trapezoid in Figure 3
        alpha = self.decay_proj_down(
            x
        )  # Down-projection (wide → narrow)
        alpha = self.decay_proj_up(
            alpha
        )  # Up-projection (narrow → wide)
        alpha = rearrange(
            alpha, "b t (h k) -> b h t k", h=self.num_heads
        )
        alpha = torch.sigmoid(alpha)  # Sigmoid activation

        # Convert to log-space for cumulative computation
        g = torch.log(alpha.clamp(min=1e-6))  # g = log(α)

        # === Process Branch 3b: β_t learning rate ===
        beta = torch.sigmoid(self.beta_proj(x))  # [B, T, H]
        beta = rearrange(beta, "b t h -> b h t", h=self.num_heads)

        # === KDA Block: Core attention computation ===
        output, final_state = chunk_kda(
            q, k, v, g, beta, self.chunk_size
        )

        # === Post-processing: Norm → Gate → Linear ===
        # Reshape for normalization
        output = rearrange(output, "b h t k -> b t h k")

        # Head-wise RMSNorm (Figure 3: "Norm" box after KDA)
        output = self.head_norm(output)

        # === Process Branch 4: Output gate (Low-rank → Sigmoid) ===
        # This is the far-right trapezoid in Figure 3
        gate = self.gate_proj_down(x)  # Down-projection
        gate = self.gate_proj_up(gate)  # Up-projection
        gate = rearrange(
            gate, "b t (h k) -> b t h k", h=self.num_heads
        )
        gate = torch.sigmoid(gate)  # Sigmoid gating (best, Table 1)

        # Apply output gate (Eq. 10)
        output = output * gate

        # Final linear projection (Figure 3: "Linear" → "Outputs")
        output = rearrange(output, "b t h k -> b t (h k)")
        output = self.o_proj(output)

        return output, final_state


class KimiLinearBlock(nn.Module):
    """
    KimiLinearBlock

    This module composes multiple KimiDeltaAttention layers (KDA, the Kimi Linear attention) followed by a
    Multi-Layer Attention (MLA) and then a Mixture-of-Experts (MoE) feed-forward block, with normalization
    and residual connections throughout.

    Architecture:
        for `kda_layers`, apply:
            - RMSNorm
            - KimiDeltaAttention
            - Residual add
            - RMSNorm
            - MoE
            - Residual add
        then:
            - RMSNorm
            - MLA
            - Residual add
            - RMSNorm
            - MoE
            - Residual add

    Args:
        dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads for KDA and MLA.
        head_dim (int): Dimension per attention head.
        chunk_size (int): Chunk size for KimiDeltaAttention.
        n_experts (int): Number of experts in the MoE layer.
        n_activated (int): Number of active experts for each input in MoE.
        expert_inter_dim (Optional[int]): Hidden dimension for each MoE expert (default: None).
        shared_expert_inter_dim (Optional[int]): Optional shared hidden dim for experts (default: None).
        use_adaptive_bias (bool): Adaptive bias flag for MoE layer.
        bias_update_rate (float): Learning rate for adaptive bias in MoE.
        kda_layers (int): Number of stacked KimiDeltaAttention (KDA) layers.
        seq_len (int): Sequence length (unused in block, for compatibility).
        additional_mla_args (dict): Additional kwargs for MLA.

    Methods:
        kda_layer(x): Applies KDA layer + MoE + residual.
        mla_layer(x, mask): Applies MLA + MoE + residual.
        forward(x, mask): Applies KDA stack then MLA layer.

    Inputs:
        x (Tensor): Input tensor of shape [B, T, dim].
        mask (Optional[Tensor]): Optional attention mask for MLA.

    Outputs:
        Tensor: Output tensor of same shape as input.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 128,
        chunk_size: int = 64,
        n_experts: int = 256,
        n_activated: int = 8,
        expert_inter_dim: Optional[int] = None,
        shared_expert_inter_dim: Optional[int] = None,
        use_adaptive_bias: bool = True,
        bias_update_rate: float = 0.01,
        kda_layers: int = 3,
        seq_len: int = 1024,
        additional_mla_args: dict = {},
    ):
        """
        Initialize KimiLinearBlock.

        See class docstring for detailed argument descriptions.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.expert_inter_dim = expert_inter_dim
        self.shared_expert_inter_dim = shared_expert_inter_dim
        self.use_adaptive_bias = use_adaptive_bias
        self.bias_update_rate = bias_update_rate
        self.kda_layers = kda_layers

        self.norm = nn.RMSNorm(dim)

        self.kda = KimiDeltaAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
        )

        self.mla = MLA(
            dim=dim, n_heads=num_heads, **additional_mla_args
        )

        self.moe = MoE(
            dim=dim,
            n_experts=n_experts,
            n_activated=n_activated,
            expert_inter_dim=expert_inter_dim,
            shared_expert_inter_dim=shared_expert_inter_dim,
        )

    def kda_layer(self, x: Tensor) -> Tensor:
        """
        Apply a single KDA (KimiDeltaAttention) block with RMSNorm, residual, and MoE.

        Args:
            x (Tensor): Input of shape [B, T, dim].

        Returns:
            Tensor: Output tensor after KDA and MoE with residual adds.
        """
        residual = x
        second = self.kda(self.norm(x))[0] + residual
        # Second Layer
        return self.moe(self.norm(second)) + second

    def mla_layer(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply MLA layer with normalization, residual, and MoE.

        Args:
            x (Tensor): Input of shape [B, T, dim].
            mask (Optional[Tensor]): Optional attention mask.

        Returns:
            Tensor: Output tensor after MLA and MoE with residual adds.
        """
        residual = x
        # First Layer
        attended = self.mla(self.norm(x)) + residual
        # Second Layer
        return self.moe(self.norm(attended)) + attended

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for KimiLinearBlock.

        Args:
            x (Tensor): Input tensor [B, T, dim].
            mask (Optional[Tensor]): Optional mask for MLA (not used by KDA).

        Returns:
            Tensor: Output tensor [B, T, dim] after sequential KDA layers, then MLA+MoE.
        """
        for i in range(self.kda_layers):
            x = self.kda_layer(x)
        return self.mla_layer(x)


class KimiLinear(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 128,
        chunk_size: int = 64,
        n_experts: int = 256,
        n_activated: int = 8,
        expert_inter_dim: Optional[int] = None,
        shared_expert_inter_dim: Optional[int] = None,
        use_adaptive_bias: bool = True,
        bias_update_rate: float = 0.01,
        kda_layers: int = 3,
        seq_len: int = 1024,
        additional_mla_args: dict = {},
        vocab_size: int = 10000,
        max_seq_length: int = 1024,
        depth: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.expert_inter_dim = expert_inter_dim
        self.shared_expert_inter_dim = shared_expert_inter_dim
        self.use_adaptive_bias = use_adaptive_bias
        self.bias_update_rate = bias_update_rate
        self.kda_layers = kda_layers
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.depth = depth

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        )

        self.norm = nn.LayerNorm(dim)

        self.blocks = nn.ModuleList(
            [
                KimiLinearBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    chunk_size=chunk_size,
                    n_experts=n_experts,
                    n_activated=n_activated,
                    expert_inter_dim=expert_inter_dim,
                    shared_expert_inter_dim=shared_expert_inter_dim,
                    use_adaptive_bias=use_adaptive_bias,
                    bias_update_rate=bias_update_rate,
                    kda_layers=kda_layers,
                    seq_len=seq_len,
                    additional_mla_args=additional_mla_args,
                )
                for _ in range(
                    depth
                )  # You can adjust the number of blocks here
            ]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
            # nn.Softmax(dim=-1),
        )

    def forward(self, x):
        normed = self.norm(self.embedding(x))

        for block in self.blocks:
            attended = block(normed)

        return self.output_head(attended)


# ################################# Kimi Linear
# if __name__ == "__main__":
#     model = KimiLinearBlock(
#         dim=512,
#         num_heads=8,
#         head_dim=64,
#         chunk_size=64,
#         n_experts=16,
#         n_activated=4,
#         kda_layers=2,
#     )

#     x = torch.randn(2, 1024, 512)

#     out = model(x)

#     print(out)
#     print(out.shape)


# if __name__ == "__main__":
#     model = KimiLinear(
#         dim=512,
#         num_heads=8,
#         head_dim=64,
#         chunk_size=64,
#         n_experts=16,
#         n_activated=4,
#         kda_layers=2,
#         depth=2,
#         vocab_size=10000,
#         seq_len=1024,
#     )

#     x = torch.randint(0, 10000, (2, 1024))

#     out = model(x)

#     print(out)
#     print(out.shape)
