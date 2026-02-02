from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy as cp

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.a3fl import A3FLTrigger


class A3FLClient(BenignClient):
    """
    A3FL-style malicious client that attacks within a specified window.
    """
    def __init__(
        self,
        selector: BaseSelector,
        trigger: A3FLTrigger,
        target_class: int,
        trigger_sample_size: int = 512,
        # --- MODIFICATION: Add attack window parameters ---
        attack_start_round: int = 1,
        attack_end_round: int = -1, # -1 means attack until the end
        poison_fraction: float = 0.1,
        malicious_epochs: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(trigger, A3FLTrigger):
            raise ValueError("A3FLClient requires an A3FLTrigger instance.")
        self.selector = selector
        self.trigger = trigger
        self.target_class = int(target_class)
        self.trigger_sample_size = int(trigger_sample_size)
        self.attack_start_round = attack_start_round
        # If end round is -1, set it to a very large number to attack indefinitely
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.poison_fraction = poison_fraction
        self.malicious_epochs = malicious_epochs

    def _build_trigger_dataloader(self) -> DataLoader:
        """Samples a small subset of local data for trigger optimization."""
        base_dataset = self.trainloader.dataset
        N = len(base_dataset)
        k = min(self.trigger_sample_size, N)
        indices = np.random.choice(np.arange(N), size=k, replace=False).tolist()
        sampled_ds = Subset(base_dataset, indices)
        batch_size = min(getattr(self.trainloader, "batch_size", 32), k)
        return DataLoader(sampled_ds, batch_size=batch_size, shuffle=True)

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        """
        Performs the A3FL attack only if the current round is within the attack window.
        Otherwise, behaves like a benign client.
        """
        # --- MODIFICATION: Check if the attack should be active ---
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            print(f"Client [{self.get_id()}]: Behaving benignly in round {round_idx} (outside attack window).")
            return super().local_train(epochs, round_idx)

        # --- Phase 1: Optimize the Trigger ---
        print(f"\n--- A3FL Client [{self.get_id()}] optimizing trigger for round {round_idx} ---")
        trigger_dl = self._build_trigger_dataloader()
        self.trigger.train_trigger(
            classifier_model=self.model,
            dataloader=trigger_dl,
            target_class=self.target_class
        )

        # --- Phase 2: Naive Training with the Optimized Trigger ---
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_class,
            poison_fraction=self.poison_fraction,
            seed=42,
            poison_exclude_target=True
            )

        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True
          )

        # Temporarily swap the trainloader and use the parent's training method
        original_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(self.malicious_epochs, round_idx)
        finally:
            self.trainloader = original_loader

        return result
    

    def aggregate_from_neighbors_attacker(self, neighbor_updates, config: Optional[Dict[str, Any]] = None, prev_global_params_per_client: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,  round_idx: int = 0,) -> Dict[str, torch.Tensor]:
      """
      Malicious aggregation routine for the attacker integrated into the client.
      - neighbor_updates: list of tuples (weights: state_dict, num_samples: int, trigger_state: any)
      - config keys:
        - attack_type: "replace" | "biased_mix" | "scale_update" | "stealthy" | "neurotoxin"
        - alpha (float): weight for attacker in "biased_mix"
        - scale (float): scaling factor for "scale_update"
        - stealth_after (int): round to begin injecting for "stealthy"
        - stealth_alpha (float): mixing weight for stealthy injection
      - prev_global_params_per_client: mapping client_id -> previous global state_dict (CPU tensors recommended)
      - returns the state_dict that the attacker will load and expose
      """

      if config is None:
        config = {}

      attack_type = config.get("attack_type", "biased_mix")
      alpha = float(config.get("biased_mix_ratio", 0.9))
      scale = float(config.get("scale", 3.0))
      stealth_after = int(config.get("stealth_after", 18))
      stealth_alpha = float(config.get("stealth_alpha", 0.7))
      neurotoxin_eta = float(config.get("neurotoxin_eta", 1.0))

      #  --- If no neighbor updates, just return own model ---
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

      else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

      # --- Load aggregated (malicious) CPU state back into model respecting device/dtype ---
      target_state = self.model.state_dict()
      for k, cpu_tensor in agg_cpu.items():
        tgt = target_state[k]
        # cast and move to target device/dtype
        target_state[k] = cpu_tensor.to(tgt.device).to(tgt.dtype)

      self.model.load_state_dict(target_state)

      # Return the state_dict attacker will expose
      #return self.model.state_dict()


