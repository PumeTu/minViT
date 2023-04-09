import torch
import torch.nn as nn
import torch.utils.data as data

class Trainer:
    """Simple boilerplate trainer class"""
    def __init__(self, model: nn.Module, optimizer: torch.optim, loss_fn: nn.Module, device: str, checkpoint: int = 0, debug: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.debug = debug
        self.checkpoint = checkpoint

    def train(self, train_loader, val_loader, epochs):
        """Train the model for number of epochs specified"""
        train_losses, val_losses, accuracies = [], [], []
        for epoch in range(epochs):
            train_loss = self.fit(train_loader)
            accuracy, val_loss = self.evaluate(val_loader) 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(accuracy)
            print(f"epoch #{epoch} | training loss: {train_loss:.4f} | validation loss: {val_loss:.4f} | accuracy: {accuracy:.2f}")
            if self.checkpoint and epoch%self.checkpoint == 0:
                print(f"Saving model checkpoint at epoch #{epoch}")
        return train_losses, val_losses, accuracies

    def fit(self, train_loader):
        """Fit the model through a single epoch"""
        self.model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            out = self.model(imgs)
            loss = self.loss_fn(out, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.debug:
            return out, total_loss / len(train_loader)
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.model(imgs)
                loss = self.loss_fn(out, labels)

                total_loss += loss.item()
                predictions = torch.argmax(out, dim=-1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(val_loader.dataset)
        return accuracy, total_loss / len(val_loader.dataset)