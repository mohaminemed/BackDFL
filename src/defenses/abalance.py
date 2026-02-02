import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List
import numpy as np

from ..fl.baseserver import FedAvgAggregator


class AdaptiveBalanceServer(FedAvgAggregator):
    """
    Adaptive BALANCE defense.
    """

    def __init__(
        self,
        model: nn.Module,
        testloader: nn.Module = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(model, testloader, device)

        self.config = config or {}

        # Parameter defaults
        self.gamma = float(self.config.get('abalance_gamma', 0.1))  # decay rate control
        self.tau = float(self.config.get('abalance_tau', 1.0))      # early benign acceptance

        # Round bookkeeping
        self.current_round: int = int(self.config.get('round', 0))
        self.total_rounds: int = int(self.config.get('num_rounds', 100))

        # Adaptive thresholds
        self.adaptive_threshold = 100.0
        self.adaptive_decay = 1.0
        self.min_required = 1
        self.relax_factor = 1.05  # fallback relaxation
        self.prev_diff_norms: Dict[int, float] = {}

        # Weighted aggregation
        self.weighted = bool(self.config.get('abalance_weighted', False))

        print(f"Initialized AdaptiveBalanceServer (tau={self.tau}, gamma={self.gamma}, weighted={self.weighted})")

    @staticmethod
    def _flatten_state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert a state dict to a single flattened vector."""
        return torch.cat([state[k].detach().cpu().flatten() for k in state], dim=0)

    # -------------------- Selection --------------------
    def accepts(self, received_state: Dict[str, torch.Tensor], reference_state: Dict[str, torch.Tensor]) -> bool:
        """Check if a client update should be accepted."""
        ref_vec = self._flatten_state_dict_to_vector(reference_state)
        recv_vec = self._flatten_state_dict_to_vector(received_state)
        diff_norm = float(torch.linalg.norm(ref_vec - recv_vec).cpu())
        effective_threshold = self.adaptive_threshold * self.adaptive_decay
        return diff_norm <= effective_threshold

    # -------------------- Aggregation --------------------
    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        """Aggregate client updates with adaptive BALANCE selection."""
        if not self.received_params:
            print("AdaptiveBalanceServer.aggregate(): warning - no updates to aggregate")
            return self.get_params()

        self.current_round = current_round
        global_state = self.get_params()
        ref_vec = self._flatten_state_dict_to_vector(global_state)

        # Compute drift norms
        diff_norms = [float(torch.linalg.norm(ref_vec - self._flatten_state_dict_to_vector(st)))
                      for st in self.received_params]

        # Median and MAD
        med = np.median(diff_norms)
        sigma = np.median(np.abs(diff_norms - med)) + 1e-8

        # Adaptive threshold shrinks over time
        self.adaptive_threshold = min(med + self.tau * sigma, self.adaptive_threshold)

        # Adaptive decay strengthens with rounds
        self.adaptive_decay = 1.0 / (1.0 + self.gamma * (self.current_round / self.total_rounds) * sigma)

        current_diff_norms = {i: dn for i, dn in enumerate(diff_norms)}

        # Initial selection
        selected_indices = [i for i, st in enumerate(self.received_params)
                            if self.accepts(st, global_state)]

        # Ensure minimum required clients
        self.min_required = max(2, int(0.3 * len(diff_norms)))

        # Fallback if too few accepted
        if len(selected_indices) < self.min_required:
            print(f"[AdaptiveBalanceServer] Fallback: relaxing threshold")
            self.adaptive_threshold *= self.relax_factor
            selected_indices = [i for i, st in enumerate(self.received_params)
                                if self.accepts(st, global_state)]

            if len(selected_indices) < self.min_required:
                # TOP-K safe selection with relative change filtering
                drifts = [(i, float(torch.linalg.norm(ref_vec - self._flatten_state_dict_to_vector(st))))
                          for i, st in enumerate(self.received_params)]
                drifts.sort(key=lambda x: x[1])
                REL_CHANGE_LIMIT = 1.5
                fallback_selected = []

                for idx, drift in drifts:
                    prev = self.prev_diff_norms.get(idx, None)
                    rel_jump = drift / (prev + 1e-8) if prev is not None else 1.0
                    if rel_jump <= REL_CHANGE_LIMIT:
                        fallback_selected.append(idx)
                    if len(fallback_selected) >= self.min_required:
                        break

                # If still not enough, take smallest drifts
                if len(fallback_selected) < self.min_required:
                    fallback_selected = [idx for idx, _ in drifts[:self.min_required]]

                selected_indices = fallback_selected
                print(f"[Fallback TOP-K] Selected updates: {selected_indices}")

        # Aggregate selected states
        selected_states = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]
        first = selected_states[0]
        averaged: Dict[str, torch.Tensor] = {}

        if self.weighted:
            total = float(sum(selected_lens))
            for k in first.keys():
                acc = sum(st[k].detach().cpu() * (l / total) for st, l in zip(selected_states, selected_lens))
                averaged[k] = acc.type(first[k].dtype)
        else:
            for k in first.keys():
                stacked = torch.stack([st[k].detach().cpu() for st in selected_states], dim=0)
                averaged[k] = torch.mean(stacked, dim=0).type(first[k].dtype)

        # Update server
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.prev_diff_norms = current_diff_norms.copy()
        self.received_params.clear()
        self.received_lens.clear()

        print(f"AdaptiveBalanceServer: aggregated {len(selected_indices)} clients (round {current_round})")
        return {k: v.cpu().clone() for k, v in averaged.items()}
