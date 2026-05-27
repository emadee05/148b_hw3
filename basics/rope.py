"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # TODO: precompute cos and sin tables of shape (max_seq_len, head_dim // 2)
        # and register them as non-persistent buffers.
        # Hint:
        #   inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        #   t = torch.arange(max_seq_len).float()
        #   freqs = torch.outer(t, inv_freq)              # (max_seq_len, head_dim // 2)
        #   self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        #   self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)


    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # TODO: implement.
        # Hint: split x into even and odd indices along head_dim, look up
        # cos/sin for the given positions, and apply the 2D rotation.
        # x: (B, num_heads, T, head_dim)
        cos = self.cos_cached[positions]  # (T, head_dim // 2)
        sin = self.sin_cached[positions]  # (T, head_dim // 2)

        # Reshape to broadcast over B and num_heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim // 2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim // 2)

        # Split into even and odd dimensions
        x_even = x[..., 0::2]  # (B, num_heads, T, head_dim // 2)
        x_odd  = x[..., 1::2]  # (B, num_heads, T, head_dim // 2)

        # Apply rotation
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos

        # Interleave back
        out = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (..., head_dim // 2, 2)
        out = out.flatten(-2)  # (B, num_heads, T, head_dim)
        return out


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        half_dim = head_dim // 2  # each of x and y gets half the head_dim
        inv_freq = base ** (-torch.arange(0, half_dim, 2).float() / half_dim)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)  # (grid_size, head_dim // 4)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    def _apply_1d_rope(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, num_heads, T, half_dim)
        cos = self.cos_cached[coords].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half_dim // 2)
        sin = self.sin_cached[coords].unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos

        out = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
        return out

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        half = self.head_dim // 2
        x_first  = x[..., :half]   # rotated by x_coords
        x_second = x[..., half:]   # rotated by y_coords

        x_first  = self._apply_1d_rope(x_first,  x_coords)
        x_second = self._apply_1d_rope(x_second, y_coords)

        return torch.cat([x_first, x_second], dim=-1)
