import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator
import numpy as np


class BalanceServer(FedAvgAggregator):
    """
    BALANCE defense adapted for server-assisted FL (and with a small helper for
    applying the same acceptance test on client-side / decentralized flows).

    Config keys (read from `config` passed at construction):
      - balance_gamma (float): gamma parameter (default 0.3)
      - balance_kappa (float): kappa parameter for exponential decay (default 1.0)
      - balance_variant (str): 'exp_decay'|'static'|'ratio' (default 'exp_decay')
            - 'exp_decay' : uses Eq.(3) with gamma * exp(-kappa * lambda(t)) * ||ref||
            - 'static'    : Variant I in paper: uses gamma * ||ref|| (no decay)
            - 'ratio'     : compare q_j = ||diff|| / ||ref|| to (gamma * decay)
      - balance_weighted (bool): if True, aggregate with sample-size weighting;
                                 if False (default), use simple arithmetic mean
                                 (paper uses arithmetic mean in DFL, so default False)
      - round (int): current round index (0-based).
      - total_rounds (int): total number of rounds T used when
                            lambda(t) = t / T is desired.

    Usage notes:
      - To use the exponential-decay variant you must provide total
        `num_rounds` in config, so the decay factor can be
        computed as exp(-kappa * lambda(t)). If not provided, lambda(t)
        defaults to the current round value (round idx) which makes the
        decay possibly very large; prefer providing total_rounds.

    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        # parameter defaults (paper uses gamma=0.3, kappa=1 in experiments).
        self.gamma = float(self.config.get('balance_gamma', self.config.get('gamma', 0.5)))
        self.kappa = float(self.config.get('balance_kappa', self.config.get('kappa', 1.0)))
        self.variant = str(self.config.get('balance_variant', 'adaptive')) # exp_decay
        self.weighted = bool(self.config.get('balance_weighted', False))

        # round bookkeeping used by lambda(t) = t / T if total_rounds is provided
        self.current_round: int = int(self.config.get('round', 0))
        self.total_rounds: int = int(self.config.get('num_rounds', 100))

        print(f"Initialized BalanceServer (gamma={self.gamma}, kappa={self.kappa}, variant={self.variant}, weighted={self.weighted})")


    @staticmethod
    def _flatten_state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
      keys = sorted(state.keys())  # enforce deterministic ordering
      parts = [state[k].detach().cpu().flatten() for k in keys]
      return torch.cat(parts, dim=0)

    @staticmethod
    def accepts(received_state: Dict[str, torch.Tensor], reference_state: Dict[str, torch.Tensor],
            gamma: float, kappa: float, current_round: int = 0, total_rounds: int = 0,
            variant: str = 'exp_decay') -> bool:
      """
        Standalone helper that implements the BALANCE acceptance test used in the
        paper (Eq. 3 and variants). This is intended to be used by client-side
        code in the decentralized flow (DFL) so clients can apply the exact same
        check when receiving neighbor models.

        Returns True if `received_state` should be accepted w.r.t `reference_state`.
      """
      # flatten state dicts
      recv_vec = BalanceServer._flatten_state_dict_to_vector(received_state)
      ref_vec = BalanceServer._flatten_state_dict_to_vector(reference_state)

      
      recv_norm = float(torch.linalg.norm(recv_vec).cpu())
      ref_norm = float(torch.linalg.norm(ref_vec).cpu())
      diff_norm = float(torch.linalg.norm(ref_vec - recv_vec).cpu())

      # compute decay
      lam = float(current_round) / float(total_rounds) if (total_rounds and total_rounds > 0) else 1.0
      decay = math.exp(-kappa * lam) if variant == 'exp_decay' else 1.0

      # fallback if reference norm is zero
      if ref_norm == 0.0:
        return diff_norm <= gamma * decay

      if variant == 'static':
        return diff_norm <= gamma * ref_norm
        
      elif variant == 'adaptive':
        diff_norm = float(torch.linalg.norm(ref_vec - recv_vec).cpu()) / ref_norm
        print(f"[BALANCE DRIFT] round {current_round}, recv_norm={recv_norm:.6f}, ref_norm={ref_norm:.6f}, "
          f"diff_norm={diff_norm:.6f}, gamma={gamma:.6f}")
        return diff_norm <= gamma 

      else:  # 'exp_decay'
        threshold = gamma * decay * ref_norm
        print(f"[BALANCE DRIFT] round {current_round}, recv_norm={recv_norm:.6f}, ref_norm={ref_norm:.6f}, "
          f"diff_norm={diff_norm:.6f}, decay={decay:.6f}, threshold={threshold:.6f}")
        return diff_norm <= threshold
    # -------------------- aggregation (server-side) --------------------
    
    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:

      global_state = self.get_params()
      num_updates = len(self.received_params)

      # Safety check
      if num_updates == 0:
        print("BalanceServer.aggregate(): warning - no updates to aggregate")
        return global_state

      # Flatten global to reference vector
      ref_vec = self._flatten_state_dict_to_vector(global_state)
      ref_norm = float(torch.linalg.norm(ref_vec)) 

      # --- Adaptive gamma update (if variant == 'adaptive') ---
      new_gamma = self.gamma
      if self.variant == 'adaptive':
        ref_norm = ref_norm + 1e-8
        diff_norms = [
            float(torch.linalg.norm(ref_vec - self._flatten_state_dict_to_vector(st))) / ref_norm
            for st in self.received_params
        ]
        if current_round < 10 :
           new_gamma = np.median(diff_norms)
        else :    
           new_gamma = min(self.gamma, np.median(diff_norms))
        print(f"[BALANCE] Gamma updated: {self.gamma:.4f} → {new_gamma:.4f}")

      # --- Client selection step ---
      selected_indices: List[int] = []
      for idx, recv_state in enumerate(self.received_params):
        if BalanceServer.accepts(
            recv_state, global_state,
            gamma=new_gamma,
            kappa=self.kappa,
            current_round=current_round,
            total_rounds=self.total_rounds,
            variant=self.variant
            ):
            selected_indices.append(idx)

      # If no one passes → fallback to gamma == Median 
      if self.variant == 'adaptive' and len(selected_indices) == 0:
         print("[BALANCE] No clients accepted → fallback to Gamma == Median.")
         new_gamma = np.median(diff_norms)
         # --- Client selection step ---
         selected_indices: List[int] = []
         for idx, recv_state in enumerate(self.received_params):
            if BalanceServer.accepts(
              recv_state, global_state,
              gamma=new_gamma,
              kappa=self.kappa,
              current_round=current_round,
              total_rounds=self.total_rounds,
              variant=self.variant
              ):
              selected_indices.append(idx)

      # If no one passes after median → fallback to fedavg         
      if self.variant == 'exp_decay' and len(selected_indices) == 0:
           print("[BALANCE] No clients accepted → fallback to Fedavg.")
           return super().aggregate()

      # Keep only selected
      selected_states = [self.received_params[i] for i in selected_indices]
      selected_lens = [self.received_lens[i] for i in selected_indices]

      # --- Aggregation ---
      averaged: Dict[str, torch.Tensor] = {}
      first = selected_states[0]

      if self.weighted:
        total = float(sum(selected_lens))
        for k in first.keys():
            acc = torch.zeros_like(first[k], dtype=torch.float32)
            for st, ln in zip(selected_states, selected_lens):
                acc += st[k].detach().cpu() * (float(ln) / total)
            averaged[k] = acc.to(first[k].dtype)
      else:
        for k in first.keys():
            stacked = torch.stack([st[k].detach().cpu() for st in selected_states], dim=0)
            averaged[k] = torch.mean(stacked, dim=0).to(first[k].dtype)

      # Update server model
      self.set_params({k: v.to(self.device) for k, v in averaged.items()})

      # Reset state buffers
      self.received_params = []
      self.received_lens = []

      print(f"[BALANCE] Aggregated {len(selected_indices)}/{num_updates} clients at round {current_round}")

      return {k: v.cpu().clone() for k, v in averaged.items()}
