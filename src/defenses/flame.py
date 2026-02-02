import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Flame requires hdbscan for clustering
try:
    import hdbscan
except ImportError:
    print("Please install hdbscan: pip install hdbscan")
    hdbscan = None

from ..fl.baseclient import BenignClient
from ..fl.baseserver import FedAvgAggregator

# robust noise generator
def _sample_noise_like(param, std_dev):
      """
      Return noise tensor with same shape/device/dtype as param and stdev std_dev.
      std_dev may be a Python float, numpy scalar, or a torch scalar/tensor.
      """
     # ensure scalar float
      try:
        std = float(std_dev)
      except Exception:
        # if std_dev is a tensor, move to CPU / convert to float safely
        std = std_dev.item() if hasattr(std_dev, 'item') else float(torch.as_tensor(std_dev))

      if std == 0.0:
        return torch.zeros_like(param)

      # create standard normal on param.device and scale by std
      noise = torch.randn(param.shape, device=param.device, dtype=param.dtype) * std
      return noise

class FlameServer(FedAvgAggregator):
    """
    Implements the FLAME defense mechanism, corrected to be compatible
    with the FedAvgAggregator base class.
    """
    def __init__(self, model: torch.nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it to use FlameServer.")

        self.config = config if config is not None else {}
        self.lamda = self.config.get('flame_lambda', 0.0001)
        self.eta = self.config.get('flame_eta', 1.0)
        
        print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}")
  
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Filters malicious updates using FLAME's clustering and then performs
        robust aggregation with clipping and adaptive noise.
        """
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()
        
        print(f"Flame detected {len(malicious_indices)} anomalous clients.")
        if malicious_indices:
            print(f"Filtering out clients at indices: {malicious_indices} and accepting clients at indices: {benign_indices}")

        if not benign_indices:
            print("Warning: Flame filtered out all clients. Global model will not be updated.")
            self.received_params.clear()
            self.received_lens.clear()
            return self.get_params()

        # --- Robust Aggregation with Clipping and Noise ---
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
        
        global_params_cpu = self.get_params()

        for original_index in benign_indices:
            local_params = self.received_params[original_index]
            
            # Use a simple average over the selected benign clients
            weight = 1.0 / len(benign_indices) 

            for name, param in local_params.items():
                if name.endswith('num_batches_tracked'): continue
                
                # Calculate the update delta (local - global)
                diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                
                # Apply clipping
                if euclidean_distances[original_index] > clip_norm:
                    diff *= clip_norm / euclidean_distances[original_index]
                
                weight_accumulator[name].add_(diff * weight)

        # Update global model parameters
        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
                if 'weight' in name or 'bias' in name:
                    std_dev = self.lamda * clip_norm
                    #noise = torch.normal(0, std_dev, param.shape, device=param.device)
                    noise = _sample_noise_like(param, std_dev)
                    param.add_(noise)
                    print(f"Added noise with std_dev={std_dev} to parameter {name}")

                    
        
        self.model.load_state_dict(final_state_dict)
        self.received_params.clear()
        self.received_lens.clear()
        
        return self.get_params()

    def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Detects anomalies using cosine similarity clustering on last layer weights.
        Returns (malicious_indices, benign_indices, all_euclidean_distances).
        """
        num_clients = len(self.received_params)
        if num_clients < 2:
            return [], list(range(num_clients)), []

        last_layer_names = self._get_last_layers(self.received_params[0])
        
        all_client_weights = []
        euclidean_distances = []
        global_params_cpu = self.get_params()

        for local_params in self.received_params:
            flat_update_diff = []
            for name, param in local_params.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update_diff)).item())

            last_layer_weights = []
            for name in last_layer_names:
                if name in local_params:
                     last_layer_weights.append(local_params[name].cpu().flatten())
            all_client_weights.append(torch.cat(last_layer_weights).numpy().astype(np.float64))
        
        # --- MODIFICATION START: Revert to using 'cosine' metric as per the reference implementation ---
        client_weights_array = np.array(all_client_weights, dtype=np.float64)

        clusterer = hdbscan.HDBSCAN(
            metric="cosine", # Use built-in cosine distance
            algorithm= "generic",
            min_cluster_size=max(2, num_clients // 2 + 1),
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(client_weights_array)
        # --- MODIFICATION END ---

        benign_indices = []
        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            else:
                 benign_indices = list(range(num_clients))

        all_indices = set(range(num_clients))
        malicious_indices = list(all_indices - set(benign_indices))

        return malicious_indices, benign_indices, euclidean_distances

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers with parameters."""
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        return param_layers[-2:]
    
    

