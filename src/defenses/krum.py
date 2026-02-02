import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator

class MKrumServer(FedAvgAggregator):
    """
    Implements the Multi-Krum defense as a subclass of FedAvgAggregator.
    """
    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        
        self.config = config if config is not None else {}
        self.num_byzantine = self.config.get('krum_f', 0)
        self.num_to_select = self.config.get('krum_m', 1)

        print(f"Initialized MKrumServer to tolerate f={self.num_byzantine} Byzantine clients and select m={self.num_to_select} for aggregation.")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Selects a subset of updates using the M-Krum algorithm and aggregates them.
        This method overrides the parent FedAvgAggregator's aggregate method.
        """
        num_updates = len(self.received_params)
        if num_updates == 0:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # M-Krum requires n > 2f + 2 to provide guarantees
        if num_updates <= 2 * self.num_byzantine + 2:
            print(f"Warning: Not enough clients ({num_updates}) for Krum with f={self.num_byzantine}. Falling back to standard FedAvg.")
            return super().aggregate()

        # --- M-Krum Algorithm ---
        # 1. We need the update deltas (local_model - global_model) for distance calculation.
        # The base server stores full models, so we calculate the deltas here.
        global_params = self.get_params() # Get current global model on CPU
        client_deltas = []
        for local_params in self.received_params:
            delta = {name: local_params[name] - global_params[name] for name in local_params}
            client_deltas.append(delta)

        flat_deltas = [torch.cat([p.flatten() for p in delta.values()]) for delta in client_deltas]

        # 2. Compute pairwise squared Euclidean distances between the deltas
        distances = torch.zeros((num_updates, num_updates))
        for i in range(num_updates):
            for j in range(i, num_updates):
                dist = torch.linalg.norm(flat_deltas[i] - flat_deltas[j]) ** 2
                distances[i, j] = distances[j, i] = dist.item()

        # 3. For each client, find the sum of distances to its k nearest neighbors
        scores = []
        # k = n - f - 2 (from the paper)
        num_neighbors = num_updates - self.num_byzantine - 2
        for i in range(num_updates):
            sorted_dists, _ = torch.sort(distances[i])
            # The first element is the distance to itself (0), so we take from 1 to num_neighbors+1
            scores.append(torch.sum(sorted_dists[1:num_neighbors+1]).item())
        
        # 4. Select the 'm' clients with the lowest scores
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:self.num_to_select]
        
        print(f"Krum selected clients at indices: {selected_indices.tolist()}")

        # 5. Aggregate only the selected clients using standard FedAvg logic
        selected_params = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]
        
        total_samples = sum(selected_lens)
        averaged = {}
        first = selected_params[0]
        for k in first.keys():
            acc = torch.zeros_like(first[k], dtype=torch.float32)
            for i, client_params in enumerate(selected_params):
                weight = float(selected_lens[i]) / float(total_samples)
                acc += client_params[k] * weight
            averaged[k] = acc

        # Load the new averaged parameters into the server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # Clear the buffers for the next round
        self.received_params = []
        self.received_lens = []
        
        return {k: v.cpu().clone() for k, v in averaged.items()}

