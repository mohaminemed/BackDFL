"""Shared malicious aggregation behavior for decentralized attacker clients.

Attacker client classes inherit AttackerAggregationMixin so the flow can route
all malicious clients through the same aggregation implementation.
"""

from typing import Any, Dict, Optional

import torch


class AttackerAggregationMixin:
    """Mixin providing a shared `aggregate_from_neighbors_attacker`."""

    def aggregate_from_neighbors_attacker(
        self,
        neighbor_updates,
        config: Optional[Dict[str, Any]] = None,
        prev_global_params_per_client: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        round_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Malicious aggregation routine shared by all attacker clients.

        - neighbor_updates: list of tuples
              (weights: state_dict, num_samples: int, metrics: any, trigger_state: any)
        - config keys:
            - attack_type: "replace" | "biased_mix" | "scale_update" |
                           "stealthy" | "neurotoxin" | "align_neighbor"
            - biased_mix_ratio (float): attacker weight for "biased_mix"
            - scale (float): scaling factor for "scale_update"
            - stealth_after (int): round to begin injecting for "stealthy"
            - stealth_ratio (float): mixing weight for stealthy injection
            - neurotoxin_eta (float): delta scaling for "neurotoxin"
            - target_neighbor_idx (int), alignment_strength (float): for
              "align_neighbor"
        - prev_global_params_per_client: mapping client_id -> previous
          global state_dict (CPU tensors recommended)
        - returns the state_dict that the attacker loads and exposes
        """

        if config is None:
            config = {}

        attack_type = config.get("attack_type", "biased_mix")
        alpha = float(config.get("biased_mix_ratio", 0.5))
        scale = float(config.get("scale", 5.0))
        stealth_after = int(config.get("stealth_after", 10))
        stealth_alpha = float(config.get("stealth_ratio", 0.7))
        neurotoxin_eta = float(config.get("neurotoxin_eta", 1.0))

        # --- If no neighbor updates, just return own model ---
        if not neighbor_updates:
            print(f"[Attacker {self.get_id()}] No neighbor updates. Keeping own model.")
            return self.model.state_dict()

        # --- Compute honest weighted average baseline (simple FedAvg) ---
        total_samples = 0
        sum_state_cpu: Dict[str, torch.Tensor] = {}
        for weights, num_samples, _, _ in neighbor_updates:
            if weights is None or num_samples is None:
                continue
            total_samples += int(num_samples)
            # ensure CPU copies for stable accumulation
            for k, v in weights.items():
                v_cpu = v.detach().cpu()
                if k not in sum_state_cpu:
                    sum_state_cpu[k] = v_cpu * float(num_samples)
                else:
                    sum_state_cpu[k] += v_cpu * float(num_samples)

        if total_samples == 0 or not sum_state_cpu:
            print(f"[Attacker {self.get_id()}] No valid neighbor updates after filtering. Keeping own model.")
            return self.model.state_dict()

        honest_state_cpu = {k: (v / float(total_samples)) for k, v in sum_state_cpu.items()}

        # --- Attacker's current state (use CPU clones for safe arithmetic) ---
        attacker_state_cpu = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        # Helper: mix two CPU state dicts with attacker weight w_att in [0,1]
        def mix_states_cpu(s_att, s_hon, w_att: float):
            mixed = {}
            for k in s_att.keys():
                mixed[k] = w_att * s_att[k] + (1.0 - w_att) * s_hon[k]
            return mixed

        # Helper: scale attacker's delta relative to honest baseline
        def scale_update_cpu(s_att, s_hon, scale_factor: float):
            out = {}
            for k in s_att.keys():
                delta = s_att[k] - s_hon[k]
                out[k] = s_hon[k] + scale_factor * delta
            return out

        # Decide strategy
        if attack_type == "replace":
            agg_cpu = attacker_state_cpu
            print(f"[Attacker {self.get_id()}] REPLACE: exposing own model.")

        elif attack_type == "biased_mix":
            agg_cpu = mix_states_cpu(attacker_state_cpu, honest_state_cpu, alpha)
            print(f"[Attacker {self.get_id()}] BIASED_MIX: alpha={alpha}.")

        elif attack_type == "scale_update":
            agg_cpu = scale_update_cpu(attacker_state_cpu, honest_state_cpu, scale)
            print(f"[Attacker {self.get_id()}] SCALE_UPDATE: scale={scale}.")

        elif attack_type == "stealthy":
            if round_idx < stealth_after:
                agg_cpu = honest_state_cpu
                print(f"[Attacker {self.get_id()}] STEALTHY: behaving honestly until round {stealth_after}.")
            else:
                agg_cpu = mix_states_cpu(attacker_state_cpu, honest_state_cpu, stealth_alpha)
                print(f"[Attacker {self.get_id()}] STEALTHY: injecting with stealth_alpha={stealth_alpha}.")

        elif attack_type == "neurotoxin":
            prev = None
            if prev_global_params_per_client is not None:
                prev = prev_global_params_per_client.get(self.get_id())

            if prev is None:
                # fallback to scaled update if no prev is available
                agg_cpu = scale_update_cpu(attacker_state_cpu, honest_state_cpu, scale)
                print(f"[Attacker {self.get_id()}] NEUROTOXIN fallback: no prev params, used scale_update (scale={scale}).")
            else:
                # Compute malicious_delta = attacker_state - prev_global for this client (assume prev in CPU)
                agg_cpu = {}
                for k in attacker_state_cpu.keys():
                    prev_k = prev.get(k)
                    if prev_k is None:
                        # fallback per-key
                        agg_cpu[k] = honest_state_cpu[k]
                        continue
                    # ensure same dtype (use float32 for safety in importance)
                    malicious_delta = attacker_state_cpu[k].to(torch.float32) - prev_k.to(torch.float32)
                    agg_cpu[k] = honest_state_cpu[k] + neurotoxin_eta * malicious_delta
                print(f"[Attacker {self.get_id()}] NEUROTOXIN: crafted update using prev_global_params (eta={neurotoxin_eta}).")

        elif attack_type == "align_neighbor":
            target_idx = int(config.get("target_neighbor_idx", 0))
            beta = float(config.get("alignment_strength", 0.8))

            # pick one neighbor
            w_t, n_t, _, _ = neighbor_updates[target_idx]

            neighbor_state_cpu = {k: v.detach().cpu() for k, v in w_t.items()}

            # align attacker toward that neighbor (or vice versa)
            agg_cpu = {}
            for k in attacker_state_cpu.keys():
                if k in neighbor_state_cpu:
                    # interpolation: attacker -> neighbor direction
                    agg_cpu[k] = (1 - beta) * attacker_state_cpu[k] + beta * neighbor_state_cpu[k]
                else:
                    agg_cpu[k] = attacker_state_cpu[k]

            print(f"[Attacker {self.get_id()}] ALIGN_NEIGHBOR: target={target_idx}, beta={beta}.")

        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

        # --- Load aggregated (malicious) CPU state back into model respecting device/dtype ---
        target_state = self.model.state_dict()
        for k, cpu_tensor in agg_cpu.items():
            tgt = target_state[k]
            # cast and move to target device/dtype
            target_state[k] = cpu_tensor.to(tgt.device).to(tgt.dtype)

        self.model.load_state_dict(target_state)

        # Return the state_dict the attacker now exposes.
        # (Previously commented out in the per-class copies -- restored
        # since the docstring and type hint both promise this value,
        # and flow.py or callers may rely on it in the future even
        # though the current dispatch call ignores the return.)
        return self.model.state_dict()