from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Set
import torch

class BasePoisoner(ABC):
    """
    Abstract base class for model poisoning strategies.
    """
    @abstractmethod
    def poison(self, model: nn.Module, dataloader: DataLoader, 
               poisoned_indices: Set[int], epochs: int, 
               learning_rate: float, device: torch.device, **kwargs) -> nn.Module:
        """
        Executes the poisoning strategy. Now accepts additional keyword arguments.
        """
        pass