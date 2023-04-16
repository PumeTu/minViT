import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from functools import partial

device = "cuda" if torch.cuda.is_available() else "cpu"

def norm_image(images, mean, std):
    return (images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

def pair(x):
    return x if isinstance(x, tuple) else (x, x)

def load_dataset(path: str = "/home/pumetu/data/",
                 dataset: str = "mnist",
                 transform: bool = False,
                 batch_size: int = 32,
                 img_size: int = 224,
                 norm: bool = True):
    """Load datasets from torchvision Image classification dataset"""
    img_size = pair(img_size)
    if dataset == "mnist":
        if transform:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize(img_size, antialias=True), 
                ])
        train_dataset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_dataset_gpu = {}
    val_dataset_gpu = {}
    train_dataset_gpu['imgaes'], train_dataset_gpu['labels'] = [item.to(device, non_blocking=True) for item in next(iter(train_loader))]
    val_dataset_gpu['imgaes'], val_dataset_gpu['labels'] = [item.to(device, non_blocking=True) for item in next(iter(val_loader))]
    if norm:
        std, mean = torch.std_mean(train_dataset_gpu['images'], dim=(0, 2, 3)) #dynamically calcualte std and mean of dataset
        norm_fn = partial(norm_image, mean=mean, std=std)
        train_dataset_gpu['images'] = norm_fn(train_dataset_gpu['images'])
        val_dataset_gpu['images'] = norm_fn(val_dataset_gpu['images'])
    dataset = {
                "train": train_dataset_gpu,
                "val": val_dataset_gpu
            }
    return dataset

