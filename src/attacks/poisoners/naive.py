import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Set

# Assuming base class is in a sibling directory
from .base import BasePoisoner

class NaivePoisoner(BasePoisoner):
    """
    Implements a naive poisoning strategy by performing standard training.

    This poisoner trains the model on the provided dataloader, which is
    expected to yield a mix of clean and poisoned samples, using a
    conventional training loop.
    """
    def poison(self, model: nn.Module, dataloader: DataLoader,
               poisoned_indices: Set[int], epochs: int,
               learning_rate: float, device: torch.device, **kwargs) -> nn.Module:

        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        print(f"Starting naive training for {epochs} epochs...")
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
        
        print("Naive poisoning training complete.")
        return model