"""
A PyTorch implementation of the Vision Transformer in the paper An Image is Worth 16x16 Words (https://arxiv.org/pdf/2010.11929.pdf)
Reference:
- An Image is Worth 16x16 Words (https://arxiv.org/pdf/2010.11929.pdf)
- Dual PatchNorm (https://arxiv.org/pdf/2302.01327.pdf)
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from dataclasses import dataclass

class PatchEmbedding(nn.Module):
    """
    Reshape 2D images (H, W, C) into a sequence of 2D patches (N, P^2C) where N = HW/P^2.
    - Note: only supports square patches
    Args:
        patch_size (int): size of each patch
        embed_dim (int): embedding dimension
        in_channels (int): number of input channels
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        patch_dim = config.patch_size**2 * config.in_channels
        self.proj = nn.Linear(patch_dim, config.embed_dim)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.norm2  = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        # split image to patches: (b, c, h, w) -> (b, n, (p^2c))
        x = rearrange(x, "b c (h hp) (w wp) -> b (h w) (hp wp c)", hp=self.patch_size, wp=self.patch_size) 
        # Dual PatchNorm proposed to add a LayerNorm both before and after the patch embedding layer: x = LN(PE(LN(x)))
        return self.norm2(self.proj(self.norm1(x))) # (b, n, embed_dim)

class SelfAttention(nn.Module):
    """Self Attention Mechanism"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.embed_dim // config.num_heads  # to keep compute and number of params constant head size is set to D / k (appendix eq. 5)
        self.qkv = nn.Linear(config.embed_dim, 3*config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1) # #(b, n, embed_dim) -> [b, n, embed_dim] * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv) # (b, num_heads, n, head_size)
        attn = (q @ k.transpose(-2, -1)) * self.head_size**-0.5 # (b, num_heads ,n, head_size) @ (b, num_heads, head_size, n) -> (b, h, n, n)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = attn @ v # (b, num_heads, n, n) @ (b, num_heads, n, head_size) -> (b, num_heads, n, head_size)
        x = rearrange(x, 'b h n d -> b n (h d)') # concat
        return self.dropout(self.proj(x)) # linear projection

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_size)
        self.fc2 = nn.Linear(config.mlp_size, config.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)

class Transformer(nn.Module):
    """
    Standard Transformer Block from the Attention is all you need paper
        Norm -> Multi-Head Attention + Residual -> Norm -> MLP + Residual
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class ViTConfig:
    img_size: int = 224
    num_classes: int = 100
    #--- Patch Embeddings ---
    in_channels: int = 3
    patch_size: int = 16
    embed_dim: int = 768
    #--- Transfomer ---------
    layers: int = 12
    mlp_size: int = 3072
    num_heads: int = 12
    dropout: float = 0.
    bias: bool = False
    pool: bool = False

class ViT(nn.Module):
    """Implementation of the Vision Transformer"""
    def __init__(self, config):
        super().__init__()
        assert config.img_size % config.patch_size == 0, "image is not divisible by patch size"
        num_patch = config.img_size**2 // config.patch_size**2
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim)) # BERT's cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch+1, config.embed_dim)) # standard learnable 1D positional embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        self.transformer = nn.ModuleList([Transformer(config) for _ in range(config.layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x) 
        cls_token = repeat(self.cls_token, "1 1 embed_dim -> b 1 embed_dim", b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1) # prepend cls embedding
        x += self.pos_embedding # add positional embedding
        x = self.dropout(x)
        for block in self.transformer:
            x = block(x)
        return self.mlp_head(x[:, 0]) # take only the cls token for classification