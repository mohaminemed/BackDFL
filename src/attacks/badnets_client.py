from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger

class BadNetsClient(AttackerAggregationMixin, BenignClient):
    """
    A malicious client specifically for the BadNets attack.

    This client uses a selector and a trigger, and implements the naive
    poisoning strategy by training on a dataloader created with the
    `make_triggered_loader` helper.
    """
    def __init__(
        self,
        # BadNets specific components
        attack_start_round: int,
        attack_end_round: int, # -1 means attack until the end
        selector: BaseSelector,
        trigger: BaseTrigger,
        target_class: int,
        poison_fraction: float = 0.1,
        malicious_epochs: int = 1,
        # Pass all BenignClient arguments via kwargs
        **kwargs,
        ):
        # Initialize the parent BenignClient with all its required arguments
        super().__init__(**kwargs)

        # Store the attack components
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class

        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.malicious_epochs = malicious_epochs
        self.poison_fraction = poison_fraction

        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_class,
            poison_fraction=self.poison_fraction,
            seed=42,
            poison_exclude_target=True
            )

        self.backdoor_trainloader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True
          )

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the training process. Behaves benignly on all rounds
        except for the specified `attack_round`.
        """
        # Check if it's the designated attack round
        if round_idx < self.attack_start_round or round_idx > self.attack_end_round:
            # If not, behave exactly like a benign client by calling the parent's method.
            print(f"--- BadnetsAttack Client [{self.get_id()}] acting benignly in round {round_idx} ---")
            return super().local_train(epochs, round_idx)

        # If it IS the attack round, proceed with the malicious logic.
        print(f"--- BadnetsAttack Client [{self.get_id()}] EXECUTING ATTACK in round {round_idx} ---")
        
        self._model.train()
        epoch_count = self.malicious_epochs
        for _ in range(epoch_count):
            for inputs, targets in self.backdoor_trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
            
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
            'round_idx': round_idx
        }
    
    

      # Return the state_dict attacker will expose
      #return self.model.state_dict()

    