import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator

class NormClippingServer(FedAvgAggregator):
    """
    Implements a robust aggregation server that clips the L2 norm of client updates.

    This version first calculates the update deltas (local_model - global_model),
    clips them, and then aggregates the clipped deltas.
    """
    def __init__(self, model: nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        
        self.config = config if config is not None else {}
        self.clipping_norm = self.config.get('clipping_norm', 5.0)
        self.mode = self.config.get('clipping_mode', 'adaptive')
        self.eta = self.config.get('eta', 1.0) # Server-side learning rate
        
        print(f"Initialized NormClippingServer with clipping_norm={self.clipping_norm}, eta={self.eta}")

    def _median_norm(self) -> float:
        """
        Computes the median L2 norm of client updates (local - global).
        Used to adaptively adjust clipping_norm.
        """
        global_params = self.get_params()

        norms = []
        for local_params in self.received_params:
            # Flatten update delta
            flat_delta = torch.cat(
                [(local_params[name] - global_params[name]).flatten() 
                 for name in local_params]
            )
            norm = torch.linalg.norm(flat_delta).item()
            norms.append(norm)

        if len(norms) == 0:
            return self.clipping_norm  # fallback, no updates

        # Median norm across clients
        median_norm = float(torch.median(torch.tensor(norms)))
        return median_norm
    

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Clips client update deltas before performing federated averaging.
        """
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # Get the current global model state on the CPU
        global_params = self.get_params()

        # Initialization Adaptive mode 
        if self.mode == 'adaptive' : 
            self.clipping_norm = self._median_norm()
            print(f"[CLIP]: Adaptive norm updated {self.clipping_norm}")

        # --- Step 1: Calculate the difference (update delta) for each client ---
        client_deltas = []
        for local_params in self.received_params:
            delta = {name: local_params[name] - global_params[name] for name in local_params}
            client_deltas.append(delta)

        # --- Step 2: Clip each client's delta based on its L2 norm ---
        for delta in client_deltas:
            # Flatten all tensors to calculate the total norm of the update
            flat_delta = torch.cat([p.flatten() for p in delta.values()])
            delta_norm = torch.linalg.norm(flat_delta)

            if delta_norm > self.clipping_norm:
                scaling_factor = self.clipping_norm / delta_norm
                for name in delta:
                    delta[name].mul_(scaling_factor)
        
        # --- Step 3: Perform federated averaging on the clipped deltas ---
        total_samples = sum(self.received_lens)
        # Accumulator for the final aggregated delta
        aggregated_delta = {name: torch.zeros_like(param) for name, param in global_params.items()}

        for i, delta in enumerate(client_deltas):
            weight_fraction = self.received_lens[i] / total_samples
            for name, param_delta in delta.items():
                if name in aggregated_delta:
                    aggregated_delta[name].add_(param_delta * weight_fraction)

        # --- Step 4: Apply the aggregated delta to the global model ---
        new_global_params = self.model.state_dict()
        for name, param in new_global_params.items():
            if name in aggregated_delta:
                # Apply update with server-side learning rate
                param.add_(aggregated_delta[name].to(self.device) * self.eta)
        
        self.model.load_state_dict(new_global_params)
        
        # Clear buffers for the next round
        self.received_params = []
        self.received_lens = []
        
        return self.get_params()


class WeakDPServer(NormClippingServer):
    """
    Implements a server that adds Gaussian noise for differential privacy.

    This defense builds upon Norm Clipping by adding noise to the final aggregated
    global model, which helps to obscure the contributions of any single client.
    """
    def __init__(self, model: nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device, config)
        
        self.std_dev = self.config.get('dp_std_dev', 0.025)
        if self.std_dev < 0:
            raise ValueError("Standard deviation for DP noise must be non-negative.")
        print(f"Initialized WeakDPServer with std_dev={self.std_dev}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        First performs clipping and aggregation, then adds Gaussian noise.
        """

        # Perform the norm clipping and aggregation from the parent class, which updates self.model
        super().aggregate()

        # Add Gaussian noise to the updated global model parameters
        with torch.no_grad():
            final_state_dict = self.model.state_dict()
            for name, param in final_state_dict.items():
                if 'weight' in name or 'bias' in name:
                    noise = torch.normal(0, self.std_dev, param.shape, device=param.device)
                    param.add_(noise)
            self.model.load_state_dict(final_state_dict)

        return self.get_params()

