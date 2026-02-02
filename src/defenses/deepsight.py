import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

try:
    import hdbscan
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    print("Please install hdbscan and scipy: pip install hdbscan scipy")
    hdbscan = None

from ..fl.baseserver import FedAvgAggregator
from .utils import NoiseDataset
from .const import NUM_CLASSES, IMG_SIZE


class DeepSightServer(FedAvgAggregator):
    """
    Implements the DeepSight defense mechanism, corrected to be compatible
    with complex model architectures like those using nn.Sequential.
    """
    def __init__(self, model: torch.nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        
        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it to use DeepSightServer.")
            
        self.config = config if config is not None else {}
        self.num_samples = self.config.get('deepsight_num_samples', 256)
        self.num_seeds = self.config.get('deepsight_num_seeds', 3)
        self.deepsight_batch_size = self.config.get('deepsight_batch_size', 64)
        self.deepsight_tau = self.config.get('deepsight_tau', 0.5)

        
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Filters malicious updates using DeepSight and then performs robust aggregation.
        """
        if not self.received_params:
            return self.get_params()
        try:
            anomalous_clients_indices, euclidean_distances = self.detect_anomalies()
            
            print(f"Deepsight detected {len(anomalous_clients_indices)} anomalous clients.")
            if anomalous_clients_indices:
                print(f"Filtering out clients at indices: {anomalous_clients_indices}")

            benign_indices = [i for i, _ in enumerate(self.received_params) if i not in anomalous_clients_indices]
            if not benign_indices:
                print("Warning: Deepsight filtered out all clients. Global model will not be updated.")
                self.received_params.clear(); self.received_lens.clear()
                return self.get_params()

            benign_distances = [euclidean_distances[i] for i in benign_indices]
            clip_norm = torch.median(torch.tensor(benign_distances)).item()

            # Perform FedAvg on the deltas of the benign updates
            global_params_cpu = self.get_params()
            aggregated_delta = {name: torch.zeros_like(param) for name, param in global_params_cpu.items()}
            total_benign_samples = sum(self.received_lens[i] for i in benign_indices)

            for i in benign_indices:
                local_params = self.received_params[i]
                num_samples = self.received_lens[i]
                weight = num_samples / total_benign_samples
                
                delta = {name: local_params[name] - global_params_cpu[name] for name in local_params}
                
                # Apply clipping to the delta
                if euclidean_distances[i] > clip_norm:
                    scaling_factor = clip_norm / euclidean_distances[i]
                    for name in delta:
                        if not name.endswith('num_batches_tracked'):
                            delta[name].mul_(scaling_factor)

                for name, param_delta in delta.items():
                    if name in aggregated_delta:
                        aggregated_delta[name].add_(param_delta, alpha=weight)

            # Apply the final aggregated delta to the global model
            new_global_state = self.model.state_dict()
            for name, param in new_global_state.items():
                if name in aggregated_delta:
                    new_global_state[name].add_(aggregated_delta[name].to(self.device))
            self.model.load_state_dict(new_global_state)

            return self.get_params()
        finally:
            # This block will always execute, ensuring cleanup happens.
            self.received_params.clear()
            self.received_lens.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("DeepSight aggregation round finished, GPU cache cleared.")
        
        

    def detect_anomalies(self) -> Tuple[List[int], List[float]]:
        """
        Orchestrates the DeepSight detection process.
        """
        num_clients = len(self.received_params)
        if num_clients < 2: return [], []

        param_names = [name for name, _ in self.model.named_parameters()]
        last_layer_weight_name = param_names[-2]
        last_layer_bias_name = param_names[-1]
        num_classes = self.model.state_dict()[last_layer_weight_name].shape[0]

        neups, TEs, euclidean_distances = self._calculate_neups(self.received_params, num_classes, last_layer_weight_name, last_layer_bias_name)
        
        classification_boundary = np.median(TEs) if TEs else 0
        te_labels = [te <= classification_boundary * 0.5 for te in TEs]

        ddifs_per_seed = self._calculate_ddifs(self.received_params)
        dist_cosine = self._calculate_cosine_distances(self.received_params, last_layer_bias_name)

        neup_clusters = hdbscan.HDBSCAN().fit_predict(neups)
        neup_dists = self._dists_from_clust(neup_clusters, num_clients)

        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(dist_cosine)
        cosine_dists = self._dists_from_clust(cosine_clusters, num_clients)
        
        ddif_dists_list = []
        for i in range(self.num_seeds):
            ddif_clusters = hdbscan.HDBSCAN().fit_predict(ddifs_per_seed[i])
            ddif_dists_list.append(self._dists_from_clust(ddif_clusters, num_clients))

        merged_ddif_dists = np.average(ddif_dists_list, axis=0)
        merged_distances = np.mean([merged_ddif_dists, neup_dists, cosine_dists], axis=0)
        final_clusters = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, allow_single_cluster=True).fit_predict(merged_distances)

        benign_clients_set, malicious_clients_set = set(), set()
        unique_clusters = [l for l in np.unique(final_clusters) if l != -1]
        
        for cluster_label in unique_clusters:
            member_indices = np.where(final_clusters == cluster_label)[0]
            num_benign = sum(1 for i in member_indices if not te_labels[i])
            if len(member_indices) > 0 and (num_benign / len(member_indices)) >= self.deepsight_tau:
                benign_clients_set.update(member_indices)
            else:
                malicious_clients_set.update(member_indices)
        
        outlier_indices = np.where(final_clusters == -1)[0]
        for i in outlier_indices:
            if not te_labels[i]: benign_clients_set.add(i)
            else: malicious_clients_set.add(i)

        return list(malicious_clients_set), euclidean_distances
    
    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        pairwise_dists = np.ones((N, N))
        for i in range(N):
            for j in range(i, N):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i, j] = pairwise_dists[j, i] = 0
        return pairwise_dists
    
    def _calculate_neups(self, local_model_updates: List[Dict[str, torch.Tensor]], num_classes: int, weight_name: str, bias_name: str) -> Tuple[np.ndarray, List[float], List[float]]:
        NEUPs, TEs, euclidean_distances = [], [], []
        global_params_cpu = self.get_params()

        for local_update in local_model_updates:
            device = local_update[weight_name].device
            flat_update = []
            for name, param in local_update.items():
                # --- FIX: Use name-based check instead of requires_grad ---
                if 'weight' in name or 'bias' in name:
                    diff = param.cpu() - global_params_cpu[name]
                    flat_update.append(diff.flatten())
            
            # Add a check for the unlikely case that a model has no trainable params
            if not flat_update:
                euclidean_distances.append(0.0)
            else:
                euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update)).item())

            global_weight = global_params_cpu[weight_name].to(device)
            global_bias = global_params_cpu[bias_name].to(device)
            diff_weight = torch.sum(torch.abs(local_update[weight_name] - global_weight), dim=1)
            diff_bias = torch.abs(local_update[bias_name] - global_bias)

            UPs_squared = (diff_bias + diff_weight) ** 2
            NEUP = UPs_squared / (torch.sum(UPs_squared) + 1e-10)
            NEUP_np = NEUP.cpu().numpy()
            NEUPs.append(NEUP_np)

            max_NEUP = np.max(NEUP_np)
            threshold = (1 / num_classes) * max_NEUP if num_classes > 0 else 0
            TEs.append(np.sum(NEUP_np >= threshold))
        return np.array(NEUPs), TEs, euclidean_distances

    def _calculate_ddifs(self, local_model_updates: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        dataset_name = self.config.get('dataset', '').upper()
        if not dataset_name or dataset_name not in NUM_CLASSES:
             raise ValueError("Dataset name must be provided in config for DeepSight")
        num_classes = NUM_CLASSES[dataset_name]; img_height, img_width, num_channels = IMG_SIZE[dataset_name]

        self.model.eval()
        local_model = copy.deepcopy(self.model)
        DDifs = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset((num_channels, img_height, img_width), self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)
            seed_ddifs = []
            for local_update in local_model_updates:
                local_model.load_state_dict(local_update)
                local_model.eval()
                DDif = torch.zeros(num_classes, device=self.device)
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        output_local = local_model(inputs)
                        output_global = self.model(inputs)
                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))
                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())
            DDifs.append(seed_ddifs)
        return np.array(DDifs)

    def _calculate_cosine_distances(self, local_model_updates: List[Dict[str, torch.Tensor]], bias_name: str) -> np.ndarray:
        N = len(local_model_updates)
        distances = np.zeros((N, N))
        global_bias = self.get_params()[bias_name]
        
        bias_diffs = [(update[bias_name].cpu() - global_bias).flatten() for update in local_model_updates]

        for i in range(N):
            for j in range(i + 1, N):
                similarity = F.cosine_similarity(bias_diffs[i], bias_diffs[j], dim=0, eps=1e-10)
                dist = 1.0 - similarity.item()
                distances[i, j] = distances[j, i] = dist
        return distances

