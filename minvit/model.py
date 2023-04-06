"""
A PyTorch implementation of the Vision Transformer in the paper An Image is Worth 16x16 Words (https://arxiv.org/pdf/2010.11929.pdf)
Reference:
- An Image is Worth 16x16 Words (https://arxiv.org/pdf/2010.11929.pdf)
- Dual PatchNorm (https://arxiv.org/pdf/2302.01327.pdf)
"""
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    """
    Reshape 2D images (H, W, C) into a sequence of 2D patches (N, P^2C) where N = HW/P^2.
    - Note: only supports square patches
    Args:
        patch_size (int): size of each patch
        embed_dim (int): embedding dimension
        in_channels (int): number of input channels
    """
    def __init__(self, patch_size:  int, embed_dim: int, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = patch_size**2 * in_channels
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.norm2  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # split image to patches: (b, c, h, w) -> (b, n, (p^2c))
        x = rearrange("b c (h hp) (w wp) -> b (h w) (hp wp c)", hp=self.patch_size, wp=self.patch_size)
        # Dual PatchNorm proposed to add a LayerNorm both before and after the patch embedding layer: x = LN(PE(LN(x)))
        x = self.norm2(self.proj(self.norm1(x))) # (b, n, embed_dim)
        return x

