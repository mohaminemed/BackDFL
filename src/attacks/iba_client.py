from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.iba import IBATrigger
import numpy as np
import copy as cp


class IBAClient(BenignClient):
    """
    A malicious client for the IBA (Irreversible Backdoor Attack).

    In each round, it first trains its U-Net trigger generator against the
    current global model, then performs naive training on its local data
    using the newly optimized generative trigger.
    """
    def __init__(
        self,
        selector: BaseSelector,
        trigger: IBATrigger,
        target_class: int,
        attack_start_round: int = 1,
        attack_end_round: int = -1,
        poison_fraction: float = 0.0,
        malicious_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(trigger, IBATrigger):
            raise ValueError("IBAClient requires an IBATrigger instance.")
        self.selector = selector
        self.trigger = trigger
        self.target_class = int(target_class)
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.poison_fraction = poison_fraction
        self.malicious_epochs = malicious_epochs

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        """
        Performs the IBA attack if within the attack window.
        """
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        try:
            # --- Phase 1: Optimize the Trigger Generator ---
            print(f"\n--- IBA Client [{self.get_id()}] optimizing U-Net generator for round {round_idx} ---")
            # The generator is trained on the full (clean) local dataset
            full_clean_loader = DataLoader(
                self.trainloader.dataset,
                batch_size=self.trainloader.batch_size,
                shuffle=True
            )
            self.trigger.train_generator(
                classifier_model=self.model,
                dataloader=full_clean_loader,
                target_class=self.target_class
            )
            # --- Phase 2: Naive Training with the Optimized Generator ---
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

            # Swap the loader and train
            original_loader = self.trainloader
            try:
               self.trainloader = poisoned_loader
               result = super().local_train(self.malicious_epochs, round_idx)
            finally:
               self.trainloader = original_loader
 
            return result

        finally:
            # This block will always execute, ensuring cleanup happens within the worker process.
            # Explicitly delete large temporary objects that might hold references.
            if 'full_clean_loader' in locals(): del full_clean_loader
            if 'poisoned_loader' in locals(): del poisoned_loader
            
            # Force PyTorch to release unused cached memory on the GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"IBAClient [{self.get_id()}] finished training, GPU cache cleared.")

        

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
      scale = float(config.get("scale", 10.0))
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


    