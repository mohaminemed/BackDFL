import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import random

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator

def _flatten_and_cat_tensors(param_dict: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Flatten and concatenate tensors from a state-dict-like mapping into a single 1D tensor on device."""
    parts = []
    for v in param_dict.values():
        parts.append(v.detach().to(device).reshape(-1))
    if parts:
        return torch.cat(parts, dim=0)
    else:
        return torch.tensor([], device=device)

def _scale_param_dict(param_dict: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    """Return a new dict with every tensor scaled in-place (non-destructive on inputs assumed)."""
    return {k: v * scale for k, v in param_dict.items()}

def _add_param_dicts(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float = 1.0) -> Dict[str, torch.Tensor]:
    """Return a new dict representing a + alpha * b (assumes same keys and shapes)."""
    return {k: (a[k] + alpha * b[k]) for k in a.keys()}

class SCCLIPServer(FedAvgAggregator):
    """
    Server-side implementation inspired by the CLIPPEDGOSSIP / SCCLIP defense.

    Behavior:
    - Compute client deltas: delta_j = local_params_j - global_params
    - Optionally bucket client deltas (average within random buckets) before clipping.
    - Compute a clipping radius tau: either fixed (config['clipping_norm']) or adaptive
      (percentile of observed norms * scale).
    - Apply CLIP(delta, tau) := min(1, tau / ||delta||_2) * delta
    - Aggregate clipped deltas weighted by client sample counts and apply with server eta.
    - Optionally apply global momentum to the aggregated delta.

    Config options (keys & defaults):
    - clipping_norm: Optional[float] = None  -> fixed tau if provided
    - adaptive: bool = True                  -> if True, compute tau from norms
    - adaptive_percentile: float = 90.0      -> percentile to use when adaptive True
    - tau_scale: float = 1.0                 -> scale factor applied to adaptive percentile
    - bucketing: bool = False                -> enable bucketing before clipping
    - bucket_size: int = 4                   -> bucket size if bucketing True
    - eta: float = 1.0                       -> server-side learning rate (apply aggregated delta * eta)
    - use_momentum: bool = False             -> whether to keep server-side momentum
    - momentum_alpha: float = 0.9            -> momentum coefficient if use_momentum True
    """

    def __init__(self, model: nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        self.config = config or {}

        # clipping / adaptive
        self.clipping_norm = self.config.get('scclip_clipping_norm', None)  # if None -> adaptive
        self.adaptive = self.config.get('scclip_adaptive', True) if self.clipping_norm is None else False
        self.adaptive_percentile = float(self.config.get('adaptive_percentile', 90.0))
        self.tau_scale = float(self.config.get('tau_scale', 1.0))

        # bucketing
        self.bucketing = bool(self.config.get('bucketing', False))
        self.bucket_size = int(self.config.get('bucket_size', 3))

        # server learning / momentum
        self.eta = float(self.config.get('eta', 1.0))
        self.use_momentum = bool(self.config.get('use_momentum', False))
        self.momentum_alpha = float(self.config.get('momentum_alpha', 0.9))
        # persistent momentum buffer (same shape as flattened model) stored as param dict matching model.state_dict()
        self._momentum_buffer: Optional[Dict[str, torch.Tensor]] = None

        print(f"Initialized SCCLIPServer(clipping_norm={self.clipping_norm}, adaptive={self.adaptive}, "
              f"adaptive_percentile={self.adaptive_percentile}, tau_scale={self.tau_scale}, bucketing={self.bucketing}, "
              f"bucket_size={self.bucket_size}, eta={self.eta}, use_momentum={self.use_momentum})")

    def _compute_client_deltas(self, global_params: Dict[str, torch.Tensor], local_params_list: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
        deltas = []
        for local in local_params_list:
            # compute local - global
            delta = {name: (local[name].detach().to(device) - global_params[name].detach().to(device)) for name in global_params.keys()}
            deltas.append(delta)
        return deltas

    def _delta_norm(self, delta: Dict[str, torch.Tensor], device: torch.device) -> float:
        flat = _flatten_and_cat_tensors(delta, device)
        if flat.numel() == 0:
            return 0.0
        return float(torch.linalg.norm(flat, ord=2).item())

    def _clip_delta(self, delta: Dict[str, torch.Tensor], tau: float, device: torch.device) -> Dict[str, torch.Tensor]:
        norm = self._delta_norm(delta, device)
        if norm == 0.0 or norm <= tau:
            # already small enough
            return delta
        scale = tau / (norm + 1e-12)
        # return scaled copy (do not mutate caller's inputs)
        return {k: (v * scale) for k, v in delta.items()}

    def _average_param_dicts(self, dicts: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Compute weighted average of param dicts. weights sum to 1.0 if provided."""
        if not dicts:
            raise ValueError("No dicts to average.")
        keys = dicts[0].keys()
        # initialize accumulator
        acc = {k: torch.zeros_like(dicts[0][k]) for k in keys}
        if weights is None:
            w = 1.0 / len(dicts)
            for d in dicts:
                for k in keys:
                    acc[k] = acc[k] + d[k] * w
        else:
            for d, wt in zip(dicts, weights):
                for k in keys:
                    acc[k] = acc[k] + d[k] * wt
        return acc

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate received client updates using SCCLIP logic.
        Returns updated global params (state_dict) after applying clipped aggregate update.
        """
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # device for computations
        device = self.device or torch.device('cpu')

        # --- Step 0: Get global params on device ---
        global_params = {k: v.detach().to(device) for k, v in self.get_params().items()}

        # --- Step 1: Compute client deltas (local - global) ---
        client_deltas = self._compute_client_deltas(global_params, self.received_params, device)

        # Optionally: bucketing (randomly shuffle and average vectors inside each bucket),
        # as described in the paper to help non-IID / heterogeneity
        if self.bucketing and len(client_deltas) > 0:
            indices = list(range(len(client_deltas)))
            random.shuffle(indices)
            buckets: List[List[Dict[str, torch.Tensor]]] = []
            for i in range(0, len(indices), self.bucket_size):
                idxs = indices[i:i + self.bucket_size]
                buckets.append([client_deltas[j] for j in idxs])
            # compute bucket means (unweighted average within bucket)
            bucketed_deltas: List[Dict[str, torch.Tensor]] = []
            for b in buckets:
                bucketed_deltas.append(self._average_param_dicts(b))
            # replace client_deltas by bucketed deltas
            client_deltas = bucketed_deltas
            # adjust received_lens to reflect bucket weighting (we'll assign equal weight within buckets)
            # For simplicity, treat each bucket as one pseudo-client with weight = sum of samples in the bucket
            # Build new received_lens accordingly:
            new_received_lens: List[int] = []
            idx_map = list(range(len(self.received_lens)))
            # compute weights per original index
            # NOTE: if bucketing changed length, we need to sum original lens for indices in each bucket
            for b in buckets:
                # we don't have indices here directly, but we can roughly approximate equal weights
                # Simpler and safe approach: set equal weight for each bucket (uniform)
                new_received_lens.append(1)
            total_bucket_weight = sum(new_received_lens)
            weights = [w / total_bucket_weight for w in new_received_lens]
        else:
            # weights based on sample counts (received_lens)
            total_samples = float(sum(self.received_lens)) if sum(self.received_lens) > 0 else 1.0
            weights = [float(l) / total_samples for l in self.received_lens]

        # --- Step 2: Determine tau (clipping radius) ---
        norms = [self._delta_norm(d, device) for d in client_deltas]
        if self.clipping_norm is not None:
            tau = float(self.clipping_norm)
        else:
            # adaptive: use percentile of norms times scale
            if len(norms) == 0:
                tau = 0.0
            else:
                # compute numpy-like percentile (torch)
                with torch.no_grad():
                    t = torch.tensor(norms, device=device)
                    # percentile via quantile
                    p = float(self.adaptive_percentile) / 100.0
                    try:
                        q = torch.quantile(t, torch.tensor(p, device=device)).item()
                    except Exception:
                        # fallback if quantile not available in this torch version
                        sorted_vals = t.sort().values
                        idx = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals))))
                        q = float(sorted_vals[idx].item())
                tau = float(max(1e-12, q * self.tau_scale))

        # ensure tau positive
        tau = max(tau, 1e-12)

        # --- Step 3: Clip each delta using CLIP(delta, tau) ---
        clipped_deltas = [self._clip_delta(d, tau, device) for d in client_deltas]

        # If bucketing was used and we replaced received_lens, weights already defined for bucket case.
        # Otherwise weights were computed from received_lens earlier.
        if not self.bucketing:
            # ensure weights length matches deltas length
            if len(weights) != len(clipped_deltas):
                # fallback to uniform weights
                weights = [1.0 / len(clipped_deltas)] * len(clipped_deltas)

        # --- Step 4: Weighted aggregation of clipped deltas ---
        # For numerical stability ensure weights sum to 1
        ws = torch.tensor(weights, dtype=torch.float32, device=device)
        if ws.sum().item() <= 0:
            ws = torch.ones_like(ws) / float(len(ws))
        else:
            ws = ws / ws.sum()

        agg_delta = {k: torch.zeros_like(v, device=device) for k, v in global_params.items()}
        for w, d in zip(ws.tolist(), clipped_deltas):
            for name in agg_delta:
                agg_delta[name] = agg_delta[name] + d[name].to(device) * float(w)

        # --- Step 5: Optional global momentum on aggregated update ---
        if self.use_momentum:
            if self._momentum_buffer is None:
                # initialize momentum buffer to zero dict
                self._momentum_buffer = {k: torch.zeros_like(v, device=device) for k, v in agg_delta.items()}
            for k in agg_delta:
                self._momentum_buffer[k] = (self.momentum_alpha * self._momentum_buffer[k] + (1.0 - self.momentum_alpha) * agg_delta[k])
            # use momentum buffer as final update
            final_delta = {k: self._momentum_buffer[k] for k in agg_delta}
        else:
            final_delta = agg_delta

        # --- Step 6: Apply aggregated update to global model with server learning rate eta ---
        new_global = {k: (global_params[k] + final_delta[k].to(device) * float(self.eta)) for k in global_params}
        print("Applying aggregated update to global model with server learning rate eta")

        # Load into model (ensure device mapping)
        # Convert tensors to CPU or to server model's device as required by self.model.load_state_dict
        model_device = next(self.model.parameters()).device if any(p.requires_grad for p in self.model.parameters()) else device
        new_global_for_load = {k: v.to(model_device) for k, v in new_global.items()}
        self.model.load_state_dict(new_global_for_load)

        # Clear buffers for the next round
        self.received_params = []
        self.received_lens = []

        return self.get_params()
