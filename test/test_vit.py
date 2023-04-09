import pytest
import torch
import torch.nn as nn
import torch.optim
from minvit.model import ViTConfig, ViT
from minvit.trainer import Trainer
from minvit.utils import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_single_mnist():
    train_loader, val_loader = load_dataset(dataset="mnist", transform=True, batch_size=32)
    config = ViTConfig(num_classes=10, in_channels=1)
    model = ViT(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device, debug=True)
    out, train_loss = trainer.fit(train_loader)

    assert out.shape == (32, 10)

