import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import numpy as np


from ..fl.baseserver import FedAvgAggregator


class SAGAServer(FedAvgAggregator):
    """
    Simple Adaptive GuArd (SAGA) defense with persistent relative-change handling.

    THIS VARIANT computes drift / relative-change **only on the last layer parameters**
    (last weight and bias parameter names) to reduce dimensionality.
    """

    def __init__(
        self,
        model: nn.Module,
        testloader: nn.Module = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, testloader, device)
        self.config = config if config is not None else {}

        # --- Parameters ---
        self.tau = float(self.config.get("saga_tau", 1.0))
        self.sigma = float(self.config.get("saga_sigma", 0.0))
        self.adaptive_threshold = float(self.config.get("adaptive_threshold", 1e9))
        self.rel_change_limit = float(self.config.get("rel_change_limit", 1.0))

        # --- Round bookkeeping ---
        self.current_round: int = int(self.config.get("round", 0))
        self.total_rounds: int = int(self.config.get("num_rounds", 100))
        self.prev_diff_norms: Dict[int, float] = {}

        # --- Persistent change tracking (per-client) ---
        # change_baselines[i] = {'baseline': float, 'since_round': int, 'consec_ok': int}
        self.change_baselines: Dict[int, Dict[str, float]] = {}
        self.max_persist_rounds = int(self.config.get("max_persist_rounds", 5))
        self.require_consecutive_ok = bool(self.config.get("require_consecutive_ok", True))
        self.consecutive_ok_needed = int(self.config.get("consecutive_ok_needed", 2))
        self._eps = float(self.config.get("eps", 1e-8))

        # --- Adaptive rel-change ---
        self.decay_floor = float(self.config.get("decay_floor", 0.5))

        # --- Weighted aggregation ---
        self.weighted = bool(self.config.get("saga_weighted", False))


        # --- Drift logging for visualization ---
        self.log_full_l2 = []       # list of lists: per-round full L2 drift
        self.log_last_l2 = []       # list of lists: per-round last-layer L2 drift
        self.log_full_cos = []      # per-round cosine sim (full)
        self.log_last_cos = []      # per-round cosine sim (last-layer)


        print(
            f"Initialized SAGAServer (tau={self.tau}, weighted={self.weighted}, adaptive_threshold={self.adaptive_threshold})"
        )

    @staticmethod
    def _flatten_state_dict_to_vector(
        state: Dict[str, torch.Tensor], keys_subset: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Flatten a state-dict into a 1D tensor. If keys_subset is provided, only those keys
        (in sorted order) are concatenated.
        """
        keys = sorted(list(state.keys()))
        if keys_subset is not None:
            keys = [k for k in keys if k in keys_subset]
        return torch.cat([state[k].detach().cpu().flatten() for k in keys], dim=0)

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """
        Return the names of the last parameter layers (those containing 'weight' or 'bias').
        Typically returns the final weight and bias parameter names (or fewer if model small).
        """
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if "weight" in name or "bias" in name]
        # return last two parameter names (or all available if fewer)
        return param_layers[-2:]

    def _effective_threshold(self) -> float:
        """
        Produce an effective drift threshold that optionally shrinks slightly over time using self.sigma.
        """
        gamma = float(self.current_round+50) / max(1.0, float(self.total_rounds+50))
        return self.adaptive_threshold + (1.0 - gamma) * (abs(self.adaptive_threshold) * self.sigma)

    def _update_rel_change_limit(self, rel_changes_current_round: List[float]) -> None:
        """
        Update the adaptive rel_change_limit using robust statistics (median + MAD).
        """
        if len(rel_changes_current_round) == 0:
            return

        vals = sorted(rel_changes_current_round)
        n = len(vals)
        # median
        median = vals[n // 2] if (n % 2 == 1) else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        # MAD
        abs_dev = sorted([abs(v - median) for v in vals])
        mad = abs_dev[n // 2] if (n % 2 == 1) else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
        tau = float(self.config.get("mad_tau", 1.0))
        mad = max(mad, self._eps)
        rel_limit = float(median + tau * mad)

        gamma = float(self.current_round+50) / max(1.0, float(self.total_rounds+50))
        decay = (1 - gamma) * (1 - self.decay_floor) + self.decay_floor
        self.rel_change_limit = decay * rel_limit

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
       return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8))
    

    # -------------------- Aggregation --------------------
    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        num_updates = len(self.received_params)
        if num_updates == 0:
            return self.get_params()

        self.current_round = current_round
        global_state = self.get_params()
        

        # Determine last-layer keys from own state (assumes all clients share same model keys)
        last_layer_names = self._get_last_layers(global_state)

        # --- Compute drift norms (L2 between global and each client state) on last-layer only ---
        ref_vec = self._flatten_state_dict_to_vector(global_state, keys_subset=last_layer_names)

        diff_norms: List[float] = []
        for st in self.received_params:
            client_vec = self._flatten_state_dict_to_vector(st, keys_subset=last_layer_names)
            diff_norms.append(float(torch.linalg.norm(ref_vec - client_vec)))

        #print(f"[SAGA] Round {current_round}: computed drift norms for {len(diff_norms)} clients.")  

        # =====================================================================
        #   REAL DRIFT LOGGING: Full parameters vs last-layer only
        # =====================================================================
        full_l2_round = []
        last_l2_round = []
        full_cos_round = []
        last_cos_round = []

        # Reference (global) vectors
        full_ref_vec = self._flatten_state_dict_to_vector(global_state)
        last_ref_vec = ref_vec   # already computed above

        for i, st in enumerate(self.received_params):
          # Flatten full client vector
          full_client_vec = self._flatten_state_dict_to_vector(st)

          # Compute L2 distances
          full_l2 = torch.linalg.norm(full_ref_vec - full_client_vec)
          last_l2 = torch.linalg.norm(last_ref_vec - self._flatten_state_dict_to_vector(st, keys_subset=last_layer_names))

          full_l2_round.append(float(full_l2))
          last_l2_round.append(float(last_l2))

          # Cosine similarities
          full_cos = self._cosine(full_ref_vec, full_client_vec)
          last_cos = self._cosine(last_ref_vec, self._flatten_state_dict_to_vector(st, keys_subset=last_layer_names))
          full_cos_round.append(full_cos)
          last_cos_round.append(last_cos)

          print(f"[SAGA][DriftLog] neighbor{i} fullL2={float(full_l2):.4f}, lastL2={float(last_l2):.4f} "
           f"| cos(full)={full_cos:.4f}, cos(last)={last_cos:.4f}")


        # Store logs
        self.log_full_l2.append(full_l2_round)
        self.log_last_l2.append(last_l2_round)
        self.log_full_cos.append(full_cos_round)
        self.log_last_cos.append(last_cos_round)

        print(f"[SAGA][DriftLog] fullL2(avg)={np.mean(full_l2_round):.4f}, lastL2(avg)={np.mean(last_l2_round):.4f} "
         f"| cos(full)={np.mean(full_cos_round):.4f}, cos(last)={np.mean(last_cos_round):.4f}")

        
        # =====================================================================
  

        # --- Adaptive threshold: median-based robust estimate ---
        med = np.median(diff_norms)
        mad = np.median(np.abs(diff_norms - med)) + 1e-8

        self.adaptive_threshold = min(med + self.tau * mad, self.adaptive_threshold)
        effective_threshold = self._effective_threshold()

        # --- Compute per-client relative changes (current round) ---
        rel_changes: List[float] = []
        for i, diff in enumerate(diff_norms):
            prev = self.prev_diff_norms.get(i)
            if prev is None:
                rel_est = abs(diff - med) / (min(diff, med) + self._eps)
                rel_changes.append(rel_est)
            else:
                rel = abs(diff - prev) / (min(prev, diff) + self._eps)
                rel_changes.append(rel)

        # --- Update adaptive relative-change limit ---
        self._update_rel_change_limit(rel_changes)

        #print(f"[SAGA] Round {current_round}: adaptive_threshold={self.adaptive_threshold:.6f}, effective_threshold={effective_threshold:.6f}, rel_change_limit={self.rel_change_limit:.6f}")

        selected_indices: List[int] = []
        log_reasons: Dict[int, str] = {}

        # --- Client selection loop (preserve original algorithmic logic) ---
        for i, diff in enumerate(diff_norms):
            baseline_entry = self.change_baselines.get(i)
            prev = self.prev_diff_norms.get(i)

            # 1) Drift threshold check
            if diff > effective_threshold:
                log_reasons[i] = (
                    f"REJECT (drift={diff:.6f} > effective_threshold={effective_threshold:.6f}; adaptive_threshold={self.adaptive_threshold:.6f})"
                )
                continue

            # 2) Persistent-change check
            if baseline_entry is not None:
                baseline = float(baseline_entry["baseline"])
                age = int(self.current_round - baseline_entry["since_round"])
                rel_to_baseline = abs(diff - baseline) / (min(diff, baseline) + self._eps)
                eff_limit = self.rel_change_limit
                if rel_to_baseline > eff_limit:
                    log_reasons[i] = (
                        f"REJECT (PERSISTED_CHANGE: rel_to_baseline={rel_to_baseline:.6f} > limit={eff_limit:.6f}; "
                        f"baseline={baseline:.6f}, curr={diff:.6f}, since={baseline_entry['since_round']})"
                    )
                    baseline_entry["consec_ok"] = 0
                    if age >= self.max_persist_rounds:
                        del self.change_baselines[i]
                        log_reasons[i] += f" | baseline aged-out after {age} rounds, released."
                    continue
                else:
                    if self.require_consecutive_ok:
                        baseline_entry["consec_ok"] = int(baseline_entry.get("consec_ok", 0)) + 1
                        if baseline_entry["consec_ok"] >= self.consecutive_ok_needed:
                            selected_indices.append(i)
                            log_reasons[i] = (
                                f"ACCEPT (persisted baseline matched: rel_to_baseline={rel_to_baseline:.6f}; released after {baseline_entry['consec_ok']} rounds)"
                            )
                            self.prev_diff_norms[i] = diff
                            del self.change_baselines[i]
                        else:
                            selected_indices.append(i)
                            log_reasons[i] = (
                                f"ACCEPT (persisted baseline matched: rel_to_baseline={rel_to_baseline:.6f}; "
                                f"consec_ok={baseline_entry['consec_ok']}/{self.consecutive_ok_needed})"
                            )
                            self.prev_diff_norms[i] = diff
                        continue
                    else:
                        selected_indices.append(i)
                        log_reasons[i] = (
                            f"ACCEPT (persisted baseline matched: rel_to_baseline={rel_to_baseline:.6f}; released)"
                        )
                        self.prev_diff_norms[i] = diff
                        del self.change_baselines[i]
                        continue

            # 3) Normal relative-change check (compare to previous round)
            if prev is not None:
                rel_change = abs(diff - prev) / (min(prev, diff) + self._eps)
                eff_limit = self.rel_change_limit
                #print(f"[SAGA] Client {i}: rel_change={rel_change:.6f} limit={eff_limit:.6f}; prev={prev:.6f}, curr={diff:.6f}")
                if rel_change > eff_limit:
                    direction = "JUMP" if diff > prev else "DROP"
                    self.change_baselines[i] = {"baseline": float(prev), "since_round": int(self.current_round), "consec_ok": 0}
                    log_reasons[i] = (
                        f"REJECT ({direction}: rel_change={rel_change:.6f} > limit={eff_limit:.6f}; prev={prev:.6f}, curr={diff:.6f}) -> persisted baseline={prev:.6f}"
                    )
                    continue
            else:
                log_reasons[i] = f"1ST Round prev=N/A, curr={diff:.6f}"

            # 4) Accept
            selected_indices.append(i)
            log_reasons[i] = f"ACCEPT (diff={diff:.6f}, prev={prev if prev is not None else 'N/A'})"
            self.prev_diff_norms[i] = float(diff)

        # --- Fallback selection logic (ensure minimum selected) ---
        min_k = max(2, int(0.3 * len(diff_norms)))
        if len(selected_indices) < min_k:
            #print(f"[SAGA] Fallback triggered: len={len(selected_indices)}, need at least {min_k}")

            drift_norm = np.array(diff_norms)
            drift_norm = (drift_norm - drift_norm.min()) / (drift_norm.max() - drift_norm.min() + self._eps)

            rel_norm = np.array(rel_changes)
            rel_norm = (rel_norm - rel_norm.min()) / (rel_norm.max() - rel_norm.min() + self._eps)

            alpha = 0.5
            combined_score = alpha * drift_norm + (1 - alpha) * rel_norm

            selected_indices = np.argsort(combined_score)[:min_k].tolist()
            #print(f"[SAGA] Fallback: selected {len(selected_indices)} (drift+stability) indices {selected_indices}")

        # --- Print per-client decisions ---
        #print("[SAGA] SELECTION report:")
        #for i in range(len(diff_norms)):
            #print(f"  Client {i:02d} → {log_reasons.get(i, 'NO ENTRY')}")

        # --- Aggregate selected updates ---
        selected_states = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]
        first = selected_states[0]

        averaged: Dict[str, torch.Tensor] = {}
        if self.weighted:
            total = float(sum(selected_lens))
            for k in first.keys():
                acc = torch.zeros_like(first[k], dtype=torch.float32)
                for st, l in zip(selected_states, selected_lens):
                    acc += st[k].detach().cpu() * (float(l) / total)
                averaged[k] = acc.type(first[k].dtype)
        else:
            for k in first.keys():
                stacked = torch.stack([st[k].detach().cpu() for st in selected_states], dim=0)
                averaged[k] = torch.mean(stacked, dim=0).type(first[k].dtype)

        # --- Update model & clear received buffers ---
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_params = []
        self.received_lens = []

        #print(f"[SAGA] Selected {len(selected_indices)} updates (round {current_round})\n")
        return {k: v.cpu().clone() for k, v in averaged.items()}
