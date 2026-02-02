import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from ..fl.baseserver import FedAvgAggregator
import numpy as np


class UBARServer(FedAvgAggregator):
    """
    UBAR defense server implementation for client-side usage. This class implements the two-stage
    UBAR filtering described in Guo et al., "Byzantine-Resilient
    Decentralized Stochastic Gradient Descent" (IEEE TCSVT, 2022).

    Important notes / expectations:
      - The training "round" is passed to aggregate(current_round).
      - The aggregator expects the following buffers to be filled by
        the caller:
          * self.received_params : List[Dict[str, Tensor]]
          * self.received_lens   : List[int] (sample counts, if weighted)
          * self.received_losses : List[float] (local loss computed by sender
            on its stochastic sample) -- **required** for Stage 2.
      - Config keys (read from `config` passed at construction):
          - ubar_rho (float): ratio 0<rho<=1 used in Stage 1 (default 0.4)
          - ubar_weighted (bool): True -> sample-size weighted aggregation
                                  False -> arithmetic mean (default False)
          - ubar_fallback (str): 'fedavg'|'best' : what to do if no
                                 candidates pass Stage 2 (default 'best')

    The aggregate() method implements exactly the two-stage UBAR
    selection: (1) choose rho*|Ni| closest neighbors (euclidean dist to
    server/global vector), (2) from those choose neighbors with loss <=
    server_loss; if none, append the best (lowest loss) from stage1.
    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None,
                 device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        self.config = config or {}
        self.rho = float(self.config.get('ubar_rho', 0.45))
        self.weighted = bool(self.config.get('ubar_weighted', False))
        self.fallback = str(self.config.get('ubar_fallback', 'best'))
        self.local_loss = 10.0

        if not hasattr(self, 'received_losses'):
            self.received_losses: List[float] = []

        print(f"Initialized UBARServer (rho={self.rho}, weighted={self.weighted}, fallback={self.fallback})")

    @staticmethod
    def _flatten_state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(state.keys())
        parts = [state[k].detach().cpu().flatten() for k in keys]
        return torch.cat(parts, dim=0)

    def _euclidean_dist(self, a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
        v_a = self._flatten_state_dict_to_vector(a)
        v_b = self._flatten_state_dict_to_vector(b)
        return float(torch.linalg.norm(v_a - v_b).cpu())

    def receive_loss(self, loss: float) -> None:
        # store params as CPU tensors to make aggregation stable
        self.received_losses.append(loss)
    
    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        """Perform UBAR aggregation for the current round.

        Expects self.received_params, self.received_lens and
        self.received_losses to be populated by the caller.
        Returns the aggregated state dict (CPU tensors).
        """
        global_state = self.get_params()
        num_updates = len(self.received_params)

        # Safety checks
        if num_updates == 0:
            print("UBAR.aggregate(): warning - no updates to aggregate")
            return global_state

        if len(self.received_losses) != num_updates:
            raise RuntimeError("UBARServer: received_losses length must match received_params length")
  
        # Stage 1: compute distances to global_state and pick rho*|Ni| closest
        distances = [self._euclidean_dist(global_state, st) for st in self.received_params]
        # compute k = max(1, floor(rho * num_updates)) to ensure at least one candidate
        k = max(1, int(math.floor(self.rho * float(num_updates))))
        idx_sorted = np.argsort(distances)
        candidate_indices = list(map(int, idx_sorted[:k]))
        print(f"UBAR: Stage 1 : selected {len(candidate_indices)}")

        # Stage 2: select those whose reported loss <= our own local loss
        if self.local_loss is None:
            self.local_loss = float(np.median([self.received_losses[i] for i in candidate_indices]))
            print("UBAR: local_loss not provided in config; using median(candidate_losses) as proxy for server_loss")
        
        print(f"UBAR: Stage 2 : recieved {len(self.received_losses)}")
        selected_indices: List[int] = [i for i in candidate_indices if self.received_losses[i] <= self.local_loss]
        print(f"UBAR: Stage 2 : selected {len(selected_indices)}")

        # If none pass Stage 2 -> append best candidate (lowest loss) as per paper
        if len(selected_indices) == 0 and len(candidate_indices) > 0:
            best_idx = min(candidate_indices, key=lambda i: self.received_losses[i])
            selected_indices.append(best_idx)
            print("[UBAR] No candidate had loss <= server_loss; appending best candidate from Stage1 (lowest loss)")

        # If still none (shouldn't happen because we forced k>=1), fallback to FedAvg
        if len(selected_indices) == 0:
            print("[UBAR] No clients selected after Stage2; falling back to FedAvg aggregation")
            return super().aggregate()

        # Keep only selected
        selected_states = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]

        # Aggregation (weighted or simple mean)
        averaged: Dict[str, torch.Tensor] = {}
        first = selected_states[0]

        if self.weighted:
            total = float(sum(selected_lens)) if sum(selected_lens) > 0 else float(len(selected_lens))
            for k_ in first.keys():
                acc = torch.zeros_like(first[k_], dtype=torch.float32)
                for st, ln in zip(selected_states, selected_lens):
                    acc += st[k_].detach().cpu() * (float(ln) / total)
                averaged[k_] = acc.to(first[k_].dtype)
        else:
            for k_ in first.keys():
                stacked = torch.stack([st[k_].detach().cpu() for st in selected_states], dim=0)
                averaged[k_] = torch.mean(stacked, dim=0).to(first[k_].dtype)

        # Update server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # Reset state buffers (caller should refill for next round)
        self.received_params = []
        self.received_lens = []
        self.received_losses = []

        print(f"[UBAR] Aggregated {len(selected_indices)}/{num_updates} clients at round {current_round} (k_stage1={k})")

        return {k: v.cpu().clone() for k, v in averaged.items()}

