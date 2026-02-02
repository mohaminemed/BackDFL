import torch
import torch.nn as nn
from typing import Dict, List, Optional

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator


class TrimmedMeanServer(FedAvgAggregator):
    """
    Implements the coordinate-wise Trimmed-Mean defense (Yin et al.) as a subclass
    of FedAvgAggregator.
    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        # explicit trimming parameter
        self.trim_b = int(self.config.get('trim_b', 1))
        print(f"Initialized TrimmedMeanServer with trim b={self.trim_b}.")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Performs coordinate-wise trimmed mean aggregation.
        Returns the new averaged parameters (state dict) on CPU.
        """
        num_updates = len(self.received_params)
        if num_updates == 0:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        b = self.trim_b
        # Trimmed-mean requires n > 2b to leave at least one value per coordinate
        if num_updates <= 2 * b:
            print(f"Warning: Not enough clients ({num_updates}) for Trimmed-Mean with b={b}. Falling back to standard FedAvg.")
            return super().aggregate()

        # Prepare averaged dict
        averaged: Dict[str, torch.Tensor] = {}

        # We iterate parameter-by-parameter. For each parameter name, we stack the
        # client tensors along a new dim 0 -> shape (num_clients, *param_shape),
        # perform a sort along dim 0, remove the smallest b and largest b entries,
        # and take the arithmetic mean of the remaining slices.

        # Use the first client's keys as canonical parameter names (assume all clients have same keys)
        first = self.received_params[0]
        param_names = list(first.keys())

        for name in param_names:
            # Collect tensors from each client for this parameter
            tensors = [client_state[name].detach().cpu() for client_state in self.received_params]

            # Stack into shape (num_clients, *param_shape)
            stacked = torch.stack(tensors, dim=0)  # dtype preserved (on CPU)

            # For coordinate-wise sorting and trimming we need to sort along dim=0
            # Sorting returns values; we don't need indices
            sorted_vals, _ = torch.sort(stacked, dim=0)

            # Trim b from both ends: keep indices [b : n - b)
            kept = sorted_vals[b:num_updates - b]

            # If nothing is left (shouldn't happen due to previous check), fallback
            if kept.numel() == 0:
                print(f"Warning: After trimming parameter {name} there are no values left. Falling back to simple mean for this parameter.")
                averaged[name] = torch.mean(stacked, dim=0)
                continue

            # Compute mean across the kept clients (dimension 0)
            averaged_param = torch.mean(kept, dim=0)

            # Cast back to original dtype if necessary (keep float32 by default)
            averaged[name] = averaged_param.type(first[name].dtype)

        # Load the new averaged parameters into the server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # Clear buffers for next round
        self.received_params = []
        self.received_lens = []

        # Return a CPU-copy of the averaged params 
        return {k: v.cpu().clone() for k, v in averaged.items()}


# MedianServer Implementation
class MedianServer(FedAvgAggregator):
    """
    Implements coordinate-wise median aggregation as a subclass of FedAvgAggregator.
    Uses the same data structure as TrimmedMeanServer (no client IDs needed).
    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        self.config = config if config is not None else {}
        print("Initialized MedianServer.")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Performs coordinate-wise median aggregation.
        Returns the new aggregated parameters (state dict) on CPU.
        """
        num_updates = len(self.received_params)
        if num_updates == 0:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # Prepare aggregated dict
        aggregated: Dict[str, torch.Tensor] = {}
        first = self.received_params[0]
        param_names = list(first.keys())

        for name in param_names:
            # Collect tensors from each client for this parameter
            tensors = [client_state[name].detach().cpu() for client_state in self.received_params]

            # Stack into shape (num_clients, *param_shape)
            stacked = torch.stack(tensors, dim=0)  # dtype preserved

            # Compute coordinate-wise median along dim=0
            median_param = torch.median(stacked, dim=0).values

            # Cast back to original dtype if necessary
            aggregated[name] = median_param.type(first[name].dtype)

        # Load the new aggregated parameters into the server model
        self.set_params({k: v.to(self.device) for k, v in aggregated.items()})

        # Clear buffers for next round
        self.received_params = []
        self.received_lens = []

        # Return a CPU-copy of the aggregated params
        return {k: v.cpu().clone() for k, v in aggregated.items()}
