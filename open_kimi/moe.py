from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Individual expert network in the MoE layer.
    Uses SwiGLU activation (SiLU-gated linear unit).
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initialize expert network.

        Args:
            dim: Input/output dimension.
            inter_dim: Intermediate/hidden dimension.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SwiGLU(x) = w2(SiLU(w1(x)) * w3(x))

        Args:
            x: Input tensor of shape (batch_size, dim).

        Returns:
            Output tensor of shape (batch_size, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SharedExpert(nn.Module):
    """
    Shared expert that processes all tokens (always active).
    Similar structure to regular expert but applied to all inputs.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initialize shared expert.

        Args:
            dim: Input/output dimension.
            inter_dim: Intermediate dimension (can be scaled for capacity).
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for shared expert.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim) or (batch_size, dim).

        Returns:
            Output tensor with same shape as input.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEGate(nn.Module):
    """
    Gating network for routing tokens to experts.
    Implements auxiliary-loss-free load balancing via adaptive biases.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int = 8,
        use_adaptive_bias: bool = True,
        bias_update_rate: float = 0.01,
    ):
        """
        Initialize MoE gate.

        Args:
            dim: Input dimension.
            n_experts: Number of routed experts.
            top_k: Number of experts to activate per token.
            use_adaptive_bias: Whether to use adaptive bias for load balancing.
            bias_update_rate: Learning rate for adaptive bias updates.
        """
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_adaptive_bias = use_adaptive_bias
        self.bias_update_rate = bias_update_rate

        # Gate weights: (n_experts, dim)
        self.gate_weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.gate_weight, std=0.02)

        # Adaptive bias for load balancing (auxiliary-loss-free)
        if use_adaptive_bias:
            self.register_buffer(
                "adaptive_bias", torch.zeros(n_experts)
            )
            self.register_buffer(
                "expert_usage", torch.zeros(n_experts)
            )
        else:
            self.register_buffer("adaptive_bias", None)
            self.register_buffer("expert_usage", None)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing scores and select top-k experts.

        Args:
            x: Input tensor of shape (batch_size, dim) or (batch_size, seq_len, dim).

        Returns:
            Tuple of (weights, expert_indices):
            - weights: Routing weights of shape (batch_size, top_k) or (batch_size, seq_len, top_k)
            - expert_indices: Selected expert indices of shape (batch_size, top_k) or (batch_size, seq_len, top_k)
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x = x.view(
                -1, dim
            )  # Flatten to (batch_size * seq_len, dim)
        else:
            batch_size, dim = x.shape
            seq_len = 1

        # Compute routing scores: (batch_size * seq_len, n_experts)
        scores = F.linear(
            x, self.gate_weight
        )  # (batch_size * seq_len, n_experts)

        # Apply adaptive bias for load balancing
        if self.use_adaptive_bias and self.adaptive_bias is not None:
            scores = scores + self.adaptive_bias.unsqueeze(0)

        # Softmax to get routing probabilities
        probs = F.softmax(scores, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            probs, self.top_k, dim=-1
        )

        # Normalize weights (optional, but helps with stability)
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Update adaptive bias based on expert usage (auxiliary-loss-free load balancing)
        if self.use_adaptive_bias and self.training:
            with torch.no_grad():
                # Count expert usage
                usage = torch.bincount(
                    top_k_indices.flatten(), minlength=self.n_experts
                ).float()
                usage = usage / (batch_size * seq_len * self.top_k)

                # Update adaptive bias to balance usage
                # If expert is overused, increase its bias (makes it less likely to be selected)
                # If expert is underused, decrease its bias (makes it more likely to be selected)
                target_usage = 1.0 / self.n_experts
                usage_diff = usage - target_usage
                self.adaptive_bias -= (
                    self.bias_update_rate * usage_diff
                )
                self.expert_usage = (
                    0.9 * self.expert_usage + 0.1 * usage
                )

        # Reshape back if needed
        if len(original_shape) == 3:
            top_k_weights = top_k_weights.view(
                batch_size, seq_len, self.top_k
            )
            top_k_indices = top_k_indices.view(
                batch_size, seq_len, self.top_k
            )

        return top_k_weights, top_k_indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) layer.
    Implements DeepSeek V3.1 style MoE with:
    - Multiple routed experts (256 by default)
    - One shared expert (always active)
    - Top-k routing (8 by default)
    - Auxiliary-loss-free load balancing via adaptive biases
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 256,
        n_activated: int = 8,
        expert_inter_dim: Optional[int] = None,
        shared_expert_inter_dim: Optional[int] = None,
        use_adaptive_bias: bool = True,
        bias_update_rate: float = 0.01,
    ):
        """
        Initialize MoE layer.

        Args:
            dim: Model dimension.
            n_experts: Number of routed experts.
            n_activated: Number of experts to activate per token (top-k).
            expert_inter_dim: Intermediate dimension for routed experts.
                If None, uses dim * 2 (typical for SwiGLU).
            shared_expert_inter_dim: Intermediate dimension for shared expert.
                If None, uses expert_inter_dim.
            use_adaptive_bias: Whether to use adaptive bias for load balancing.
            bias_update_rate: Learning rate for adaptive bias updates.
        """
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_activated = n_activated

        # Set intermediate dimensions
        if expert_inter_dim is None:
            expert_inter_dim = dim * 2
        if shared_expert_inter_dim is None:
            shared_expert_inter_dim = expert_inter_dim

        self.expert_inter_dim = expert_inter_dim
        self.shared_expert_inter_dim = shared_expert_inter_dim

        # Gate for routing
        self.gate = MoEGate(
            dim=dim,
            n_experts=n_experts,
            top_k=n_activated,
            use_adaptive_bias=use_adaptive_bias,
            bias_update_rate=bias_update_rate,
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [Expert(dim, expert_inter_dim) for _ in range(n_experts)]
        )

        # Shared expert (always active)
        self.shared_expert = SharedExpert(
            dim, shared_expert_inter_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        original_shape = x.shape
        batch_size, seq_len, dim = x.shape

        # Flatten for processing
        x_flat = x.view(-1, dim)  # (batch_size * seq_len, dim)

        # Get routing weights and expert indices
        weights, expert_indices = self.gate(
            x_flat
        )  # (batch_size * seq_len, top_k)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_id in range(self.n_experts):
            # Find tokens routed to this expert
            mask = (
                expert_indices == expert_id
            )  # (batch_size * seq_len, top_k)

            if not mask.any():
                continue

            # Get the weight index for this expert
            weight_idx = torch.nonzero(mask, as_tuple=False)
            if len(weight_idx) == 0:
                continue

            # Get tokens and their corresponding weights
            token_indices = weight_idx[:, 0]
            weight_positions = weight_idx[:, 1]

            # Get unique token indices and aggregate weights
            unique_tokens, inverse_indices = torch.unique(
                token_indices, return_inverse=True
            )
            expert_input = x_flat[unique_tokens]

            # Compute expert output
            expert_output = self.experts[expert_id](expert_input)

            # Aggregate weights for tokens that route to this expert multiple times
            token_weights = weights[token_indices, weight_positions]
            aggregated_weights = torch.zeros(
                len(unique_tokens), device=x.device, dtype=x.dtype
            )
            aggregated_weights.scatter_add_(
                0, inverse_indices, token_weights
            )

            # Apply weights and accumulate
            output[
                unique_tokens
            ] += expert_output * aggregated_weights.unsqueeze(-1)

        # Apply shared expert (always active for all tokens)
        shared_output = self.shared_expert(x_flat)
        output = output + shared_output

        # Reshape back
        output = output.view(original_shape)
        return output


# if __name__ == "__main__":
#     # Example usage
#     torch.manual_seed(42)

#     # Model parameters (DeepSeek V3.1 style)
#     dim = 2048
#     n_experts = 256
#     n_activated = 8
#     batch_size = 2
#     seq_len = 128

#     # Create MoE layer
#     moe = MoE(
#         dim=dim,
#         n_experts=n_experts,
#         n_activated=n_activated,
#         use_adaptive_bias=True
#     )

#     # Create dummy input
#     x = torch.randn(batch_size, seq_len, dim)

#     # Forward pass
#     output = moe(x)

#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Number of experts: {n_experts}")
#     print(f"Activated experts per token: {n_activated}")
#     print(f"MoE layer created successfully!")

#     # Check expert usage statistics
#     if moe.gate.use_adaptive_bias:
#         print(f"\nExpert usage statistics:")
#         print(f"Adaptive bias range: [{moe.gate.adaptive_bias.min():.4f}, {moe.gate.adaptive_bias.max():.4f}]")
#         print(f"Expert usage range: [{moe.gate.expert_usage.min():.4f}, {moe.gate.expert_usage.max():.4f}]")
