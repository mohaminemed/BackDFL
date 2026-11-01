from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.iba import IBATrigger
import numpy as np
import copy as cp


class IBAClient(AttackerAggregationMixin, BenignClient):
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
        poison_fraction: float = 0.25,
        malicious_epochs: int = 5,
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

        


      # Return the state_dict attacker will expose
      #return self.model.state_dict()

class IBAGeneratorOnlyClient(IBAClient):
    """
    Control for IBA: the U-Net trigger generator is optimized against the current model exactly as in
    IBAClient, but the classifier is trained on CLEAN local data only (no poisoned samples, no target label).
    Any ASR measured with this trigger is therefore the transfer / "adversarial" baseline of the generator,
    not an injected backdoor.
    """
    def __init__(self, *args, generator_after_training: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # False: generator optimized on the model received at the start of the round (same order as IBAClient)
        # True : generator optimized on the model I am about to send (trigger tuned to the exposed weights)
        self.generator_after_training = generator_after_training

    def _train_generator(self):
        full_clean_loader = DataLoader(
            self.trainloader.dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
        )
        try:
            self.trigger.train_generator(
                classifier_model=self.model,
                dataloader=full_clean_loader,
                target_class=self.target_class,
            )
        finally:
            del full_clean_loader

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super(IBAClient, self).local_train(epochs, round_idx)
        try:
            print(f"\n--- IBA-GenOnly Client [{self.get_id()}] round {round_idx}: "
                  f"generator optimized, NO poisoning ---")
            if not self.generator_after_training:
                self._train_generator()

            # clean training with the same number of epochs as a benign client
            result = super(IBAClient, self).local_train(epochs, round_idx)

            if self.generator_after_training:
                self._train_generator()
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"IBAGeneratorOnlyClient [{self.get_id()}] finished, GPU cache cleared.")