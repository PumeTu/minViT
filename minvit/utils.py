import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(path: str = "/home/pumetu/data/", dataset: str = "mnist", transform: bool = False, batch_size: int =32):
    if dataset == "mnist":
        if transform:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((224, 224), antialias=True), 
                transforms.Normalize((0.1307,),(0.3081,))
                ])
        train_dataset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader