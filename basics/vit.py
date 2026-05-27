"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.model import Block, Head
from basics.rope import RoPE1D, RoPE2D


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def _apply_rope_to_attention_heads(model: nn.Module, head_dim: int, seq_len: int) -> None:
    """
    Attach RoPE1D modules to every basics.model.Head and patch its forward.

    This assumes each Head has:
      q_proj, k_proj, v_proj, dropout
    like your existing basics.model.Head implementation.
    """
    from basics.model import Head

    for module in model.modules():
        if not isinstance(module, Head):
            continue

        module.rope = RoPE1D(head_dim=head_dim, max_seq_len=seq_len)

        def rope_forward(self, x: torch.Tensor):
            B, T, C = x.shape

            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Convert to fake num_heads=1 shape for RoPE1D:
            # (B, 1, T, head_dim)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)

            positions = torch.arange(T, device=x.device)
            q = self.rope(q, positions).squeeze(1)
            k = self.rope(k, positions).squeeze(1)

            wei = q @ k.transpose(-2, -1) * (C ** -0.5)

            # Preserve the existing behavior for encoder heads.
            if hasattr(self, "tril"):
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

            wei = torch.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            out = wei @ v
            return out

        module.forward = rope_forward.__get__(module, module.__class__)

def _apply_rope2d_to_attention_heads(
    model: nn.Module,
    head_dim: int,
    grid_size: int,
) -> None:
    """
    Attach RoPE2D modules to every basics.model.Head and patch its forward.

    Sequence layout:
      token 0 = CLS token
      tokens 1...N = image patches in flattened row-major order

    For CLS, we assign coordinate (0, 0).
    For patch i, coordinates are:
      x = column index
      y = row index
    """

    for module in model.modules():
        if not isinstance(module, Head):
            continue

        module.rope2d = RoPE2D(head_dim=head_dim, grid_size=grid_size)

        def rope2d_forward(self, x: torch.Tensor):
            B, T, C_in = x.shape

            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            head_dim = q.shape[-1]

            # Build 2D coordinates for [CLS] + flattened patches.
            # CLS gets (0, 0).
            num_patch_tokens = T - 1
            grid = int(num_patch_tokens ** 0.5)

            patch_ids = torch.arange(num_patch_tokens, device=x.device)
            y_patch = patch_ids // grid
            x_patch = patch_ids % grid

            x_coords = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=x.device), x_patch.long()],
                dim=0,
            )
            y_coords = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=x.device), y_patch.long()],
                dim=0,
            )

            q = q.unsqueeze(1)  # (B, 1, T, head_dim)
            k = k.unsqueeze(1)

            q = self.rope2d(q, x_coords=x_coords, y_coords=y_coords).squeeze(1)
            k = self.rope2d(k, x_coords=x_coords, y_coords=y_coords).squeeze(1)

            wei = q @ k.transpose(-2, -1) * (head_dim ** -0.5)

            if hasattr(self, "tril"):
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

            wei = torch.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            out = wei @ v
            return out

        module.forward = rope2d_forward.__get__(module, module.__class__)

class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), you may want a `return_all_tokens=True` flag that returns the
    full (B, num_patches+1, d_model) sequence instead. Add it when you get there.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str="learned",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        seq_len = num_patches + 1

        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = pos_encoding
        if pos_encoding == "learned":
          self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        else:
          self.pos_embed = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    block_size=seq_len,
                    is_decoder=False,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        if pos_encoding == "rope":
            head_dim = d_model // num_heads
            _apply_rope_to_attention_heads(self, head_dim=head_dim, seq_len=seq_len)
        elif pos_encoding == "rope2d":
            head_dim = d_model // num_heads
            grid_size = img_size // patch_size
            _apply_rope2d_to_attention_heads(
                self,
                head_dim=head_dim,
                grid_size=grid_size,
            )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.pos_encoding=="learned":
          x = x + self.pos_embed


        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if return_all_tokens:
            return x

        return x[:, 0, :]