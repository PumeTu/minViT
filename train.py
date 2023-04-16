import argparse

import torch
import torch.optim as optim

from minvit.trainer import Trainer
from minvit.model import ViT, ViTConfig
from minvit.utils import load_dataset

# --------------------- Hyper Params --------------------------
lr = 0.01
epochs = 10
dataset = "mnist"
path = "/home/pumetu/data/"
transform = True
batch_size = 64
norm = True
# ViTConfig
img_size = 224
num_classes = 10
in_channels = 1
patch_size = 16
embed_dim = 768
layers = 12
mlp_size = 3072
num_heads = 12
dropout = 0.
bias = False
pool = False
# --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
config = ViTConfig(
    img_size=img_size,
    num_classes=num_classes,
    in_channels=in_channels,
    patch_size=patch_size,
    embed_dim=embed_dim,
    layers=layers,
    mlp_size=mlp_size,
    num_heads=num_heads,
    dropout=dropout,
    bias=bias
    pool=pool
)
dataset = load_dataset(
    path=path,
    dataset=dataset,
    transform=transform,
    batch_size=batch_size,
    img_size=img_size,
    norm=norm
)