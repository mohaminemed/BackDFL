import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Set
import copy
from .base import BasePoisoner

class ScalingPoisoner(BasePoisoner):
    """
    Implements the model update scaling attack.

    This poisoner first trains the model conventionally on the poisoned data,
    then scales the difference between the initial and final weights by a
    large factor.
    """
    def __init__(self, scale_factor: float = 10.0):
        """
        Initializes the ScalingPoisoner.

        Args:
            scale_factor (float): The factor by which to multiply the model
                                  update (e.g., 10.0).
        """
        self.scale_factor = scale_factor
        # print(f"ScalingPoisoner initialized with scale_factor: {self.scale_factor}")

    def poison(self, model: nn.Module, dataloader: DataLoader, 
               poisoned_indices: Set[int], epochs: int, 
               learning_rate: float, device: torch.device) -> nn.Module:
        
        model.to(device)
        initial_state_dict = copy.deepcopy(model.state_dict())
        
        # --- Step 1: Naive training on the full (mixed) dataset ---
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
        
        # print("Naive training complete. Now scaling the update.")
        
        # --- Step 2: Calculate and apply the scaled update ---
        final_state_dict = model.state_dict()
        scaled_state_dict = initial_state_dict.copy()

        for key in initial_state_dict:
            update = final_state_dict[key] - initial_state_dict[key]
            scaled_update = self.scale_factor * update
            scaled_state_dict[key] += scaled_update
        
        model.load_state_dict(scaled_state_dict)
        # print("Model update scaled and applied.")
        
        return model