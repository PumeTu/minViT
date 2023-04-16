import pytest
import torch
import torch.nn as nn
import torch.optim
from minvit.model import ViTConfig, ViT
from minvit.trainer import Trainer
from minvit.utils import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_single_mnist():
    train_loader, _ = load_dataset(dataset="mnist", transform=True, batch_size=64)
    config = ViTConfig(num_classes=10, in_channels=1)
    model = ViT(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device, debug=True)
    out, _ = trainer.fit(train_loader)
    assert out.shape == (32, 10)

def test_mnist():
    train_loader, val_loader = load_dataset(dataset="mnist", transform=True, batch_size=64)
    config = ViTConfig(num_classes=10, in_channels=1)
    model = ViT(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device)
    train_loss, val_loss, accuracy = trainer.train(train_loader, val_loader, epochs=2)
    print(f"train: {train_loss}, val: {val_loss}, accuracy: {accuracy}")
