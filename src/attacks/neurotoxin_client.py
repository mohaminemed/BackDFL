import torch
from typing import Dict, Any, Optional
import copy
from torch.utils.data import DataLoader
from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger

class NeurotoxinClient(AttackerAggregationMixin, BenignClient):
    """
    An implementation of the Neurotoxin attack, aligned with the
    paper's formal algorithm.

    This version uses the aggregated global update from the previous round to
    identify and mask important parameters, making the attack more robust.
    """
    def __init__(
        self,
        selector: BaseSelector,
        trigger: BaseTrigger,
        target_class: int,
        attack_start_round: int,
        attack_end_round: int = -1,
        mask_k_percent: float = 0.05, # Mask the top 5% of parameters
        scale_factor: float = 1.5,
        poison_fraction: float = 0.1,
        malicious_epochs: int = 1,
        **kwargs,
     ):
        super().__init__(**kwargs)
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.mask_k_percent = mask_k_percent
        self.scale_factor = scale_factor
        self.poison_fraction = poison_fraction
        self.malicious_epochs = malicious_epochs
        self.attack_mode = "continuous" #"sporadic" #
        self.sporadic_k = 2
        self.sporadic_p = 2

        # Precompute sporadic attack rounds
        if self.attack_mode.lower() == "sporadic":
          self.generate_sporadic_schedule(self.sporadic_k, self.sporadic_p)
        else:
          self.attack_rounds = None
    
    def generate_sporadic_schedule(self, k:int, p:int):
        """
        Generate the attack schedule for sporadic mode.
    
        Parameters:
          k (int): number of consecutive attack rounds
          p (int): number of consecutive benign rounds
        
        Produces:
          self.attack_rounds: a set of round indices during which 
                            the client should perform the attack.
        """
        attack_rounds = set()
        start = self.attack_start_round
        end = int(self.attack_end_round)

        current = start
        while current <= end:
          # Add k attack rounds
          for r in range(current, min(current + k, end + 1)):
            attack_rounds.add(r)
          # Skip p benign rounds
          current += k + p
        print(f"Attack rounds: {attack_rounds}")
        self.attack_rounds = attack_rounds

    def is_attack_round(self, round_number: int) -> bool:
       if self.attack_mode.lower() == "continuous":
         return self.attack_start_round <= round_number <= self.attack_end_round
       elif self.attack_mode.lower() == "sporadic":
         return round_number in self.attack_rounds
       return False
    

    def local_train(self, epochs: int, round_idx: int, prev_global_grad: Dict[str, torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """
        Neurotoxin local training with robust importance masking.

        - Builds importance = |delta| / (|param| + eps) using `prev_global_grad` (if provided).
        - Selects top-k important parameters (mask_k_percent) and ZEROes gradients for those
          parameters during local poisoned training; keeps gradients for the rest.
        - Uses device/dtype-safe conversions and prints debugging info.
        """
        # If outside attack window, behave like a benign client
        if not self.is_attack_round(round_idx):
            print(f"\n--- Neurotoxin Client [{self.get_id()}] behaving benignly for round {round_idx} ---")
            return super().local_train(epochs, round_idx)

        print(f"\n--- Neurotoxin Client [{self.get_id()}] starting hybrid attack for round {round_idx} ---")

        # keep a CPU copy of initial state (for optional scaling later)
        initial_state_cpu = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        # ---------- Build robust grad mask (top-k by normalized importance) ----------
        grad_mask: Optional[Dict[str, torch.Tensor]] = None
        if prev_global_grad is None:
            print(f"Client [{self.get_id()}]: No previous global gradient. Attacking without mask.")
        else:
            # Only consider keys corresponding to model.named_parameters()
            model_param_keys = set(name for name, _ in self.model.named_parameters())
            # Prepare containers
            importance_parts = []
            key_to_delta = {}
            eps = 1e-12

            # Move deltas to CPU float32 and compute per-key importance
            for name, delta in prev_global_grad.items():
                if name not in model_param_keys:
                    continue
                d_cpu = delta.detach().cpu().to(torch.float32)
                # Use current parameter value from model state dict for normalization
                param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                # importance = |delta| / (|param| + eps)
                importance = (d_cpu.abs() / (param_cpu.abs() + eps)).flatten()
                importance_parts.append(importance)
                key_to_delta[name] = d_cpu  # keep delta (cpu float32)

            if len(importance_parts) == 0:
                print(f"Client [{self.get_id()}]: No matching trainable keys in prev_global_grad. Attacking without mask.")
            else:
                all_importances = torch.cat(importance_parts)
                num_params = all_importances.numel()
                k = max(1, int(self.mask_k_percent * num_params))
                # if mask_k_percent small and distribution has many zeros, topk still works
                k = min(k, num_params)

                # compute top-k threshold on importance
                topk_vals, _ = torch.topk(all_importances, k, largest=True, sorted=True)
                threshold = topk_vals[-1].item()
                print(f"Client [{self.get_id()}]: importance threshold (top {self.mask_k_percent*100:.2f}%) = {threshold:.6e}, k={k}")

                # Build boolean mask per key: True -> KEEP gradient (unimportant), False -> ZERO OUT (important)
                grad_mask = {}
                total_kept = 0
                total_params = 0
                for name, delta_cpu in key_to_delta.items():
                    param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                    importance_key = (delta_cpu.abs() / (param_cpu.abs() + eps))
                    mask_key = (importance_key < threshold)  # True = unimportant -> keep
                    grad_mask[name] = mask_key  # boolean tensor on CPU
                    total_kept += int(mask_key.sum().item())
                    total_params += mask_key.numel()

                print(f"Client [{self.get_id()}]: mask built. total_params={total_params}, kept(unimportant)={total_kept}, zeroed(important)={total_params - total_kept}")

        # ---------- Create poisoned dataloader ----------
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

        # ---------- Local poisoned training with mask applied to grads ----------
        self.model.train()
        for _ in range(self.malicious_epochs):
            for data, target in poisoned_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                # Apply mask: convert mask to grad dtype/device before multiplying
                if grad_mask is not None:
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                continue
                            if name in grad_mask:
                                mask_cpu = grad_mask[name]
                                # convert mask to param.grad dtype & device
                                mask = mask_cpu.to(param.grad.dtype).to(param.grad.device)
                                param.grad.mul_(mask)

                self.optimizer.step()

        # ---------- Optional model scaling (apply on CPU for safety) ----------
        if self.scale_factor > 1.0:
            print(f"Client [{self.get_id()}]: Applying scale factor of {self.scale_factor:.2f}")
            final_state_cpu = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            scaled_state = {}
            for k in initial_state_cpu:
                update = final_state_cpu[k] - initial_state_cpu[k]
                scaled_state[k] = initial_state_cpu[k] + (self.scale_factor * update)
            # load scaled state back onto model (respecting device/dtype)
            # we move each tensor to model param's device/dtype
            target_state = self.model.state_dict()
            for k, tensor_cpu in scaled_state.items():
                tgt = target_state[k]
                target_state[k] = tensor_cpu.to(tgt.device).to(tgt.dtype)
            self.model.load_state_dict(target_state)

        # step the scheduler if present
        if self.scheduler:
            self.scheduler.step()

        # Evaluate locally and prepare return payload
        # metrics = self.local_evaluate()['metrics']
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
            'round_idx': round_idx
        }


      # Return the state_dict attacker will expose
      #return self.model.state_dict()

