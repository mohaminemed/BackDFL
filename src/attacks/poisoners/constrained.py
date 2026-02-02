import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Set, Dict
from .base import BasePoisoner

class ConstrainedPoisoner(BasePoisoner):
    """
    Implements the Neurotoxin constrained poisoning attack.

    This strategy makes backdoors durable by only updating parameters that are
    infrequently changed by benign clients. Importance is determined from the
    previous round's aggregated global gradient.
    """
    def __init__(self, mask_k_percent: float = 0.05):
        """
        Initializes the ConstrainedPoisoner.

        Args:
            mask_k_percent (float): The top-k percentage of gradients to treat
                                    as important and therefore mask (e.g., 0.05 for 5%).
        """
        # Note: The paper uses bottom-k% for the constraint set, which is
        # equivalent to masking the top-k%. k=5% means we mask the top 5%
        # and update the bottom 95%.
        if not 0.0 <= mask_k_percent <= 1.0:
            raise ValueError("mask_k_percent must be between 0.0 and 1.0.")
        self.mask_k_percent = mask_k_percent
        print(f"ConstrainedPoisoner (Neurotoxin) initialized. Masking top {self.mask_k_percent*100:.2f}% of gradients.")

    def poison(self, model: nn.Module, dataloader: DataLoader, 
               poisoned_indices: Set[int], epochs: int, 
               learning_rate: float, device: torch.device, **kwargs) -> nn.Module:
        
        prev_global_grad = kwargs.get('prev_global_grad')
        if prev_global_grad is None:
            raise ValueError("Neurotoxin requires 'prev_global_grad' from the previous round.")

        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # --- Step 1: Determine importance mask from previous global gradient ---
        # This is done once before training begins.
        all_grads = torch.cat([v.flatten().abs() for v in prev_global_grad.values()])
        # We find the threshold for the top-k% of gradients
        threshold = torch.quantile(all_grads, 1.0 - self.mask_k_percent)
        
        importance_mask = {
            name: (grad.abs() >= threshold)
            for name, grad in prev_global_grad.items()
        }
        print(f"Calculated importance threshold: {threshold.item():.4f}")

        # --- Step 2: Train on the (mixed) poisoned dataset ---
        # Neurotoxin trains on the client's full dataset, where some samples
        # have been poisoned by the trigger and relabeled.
        model.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # --- Step 3: Apply constraint (gradient projection) ---
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in importance_mask:
                            # Zero out the gradients for the "important" parameters
                            param.grad[importance_mask[name]] = 0.0
                
                optimizer.step()

        print("Constrained (Neurotoxin) poisoning complete.")
        return model