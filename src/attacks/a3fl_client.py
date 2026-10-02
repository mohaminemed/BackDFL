from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy as cp

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.a3fl import A3FLTrigger


class A3FLClient(AttackerAggregationMixin, BenignClient):
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
    


      # Return the state_dict attacker will expose
      #return self.model.state_dict()


