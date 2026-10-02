from typing import Any, Dict, Optional
import copy
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger
import copy as cp 

class ScalingAttackClient(AttackerAggregationMixin, BenignClient):
    """
    A malicious client for the Model Scaling attack, updated to be more
    accurate to the original paper's description.
    """
    def __init__(
        self,
        # New parameters for single-shot attack and dynamic scaling
        attack_start_round: int,
        attack_end_round: int, # -1 means attack until the end
        num_total_clients: int,
        num_malicious_clients: int,
        malicious_epochs: int,
        # Attack specific components
        selector: BaseSelector,
        trigger: BaseTrigger,
        target_class: int,
        scale_factor: Optional[float] = None, 
        poison_fraction: float = 0.1,

        # Pass all BenignClient arguments via kwargs
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class
        
        # Store new parameters
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.scale_factor = scale_factor
        self.num_total_clients = num_total_clients
        self.num_malicious_clients = num_malicious_clients
        self.malicious_epochs = malicious_epochs
        self.poison_fraction = poison_fraction
        
        poisoned_dataset = BackdoorDataset(
            original_dataset=cp.deepcopy(self.trainloader.dataset),
            trigger_fn=self.trigger.apply,
            target_label=self.target_class,
            poison_fraction=self.poison_fraction,
            seed=42,
            poison_exclude_target=True
            )
        self.backdoor_trainloader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True)

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the training process. Behaves benignly on all rounds
        except for the specified `attack_round`.
        """
        # Check if it's the designated attack round
        if round_idx < self.attack_start_round or round_idx > self.attack_end_round:
            # If not, behave exactly like a benign client by calling the parent's method.
            print(f"--- ScalingAttack Client [{self.get_id()}] acting benignly in round {round_idx} ---")
            return super().local_train(epochs, round_idx)

        # If it IS the attack round, proceed with the malicious logic.
        print(f"--- ScalingAttack Client [{self.get_id()}] EXECUTING ATTACK in round {round_idx} ---")
        
        initial_state_dict = copy.deepcopy(self.model.state_dict())

        self._model.train()
        for _ in range(self.malicious_epochs):
            for inputs, targets in self.backdoor_trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
            
        # Determine the scale factor dynamically if not provided
        effective_scale_factor = self.scale_factor
        if effective_scale_factor is None:
            # As per the paper, scale by N/η. Using total clients as N.
            effective_scale_factor = self.num_total_clients / self.num_malicious_clients
        
        print(f"Client [{self.get_id()}]: Using scale factor: {effective_scale_factor:.2f}")

        final_state_dict = self.model.state_dict()
        scaled_state_dict = initial_state_dict.copy()

        for key in initial_state_dict:
            update = final_state_dict[key] - initial_state_dict[key]
            scaled_update = effective_scale_factor * update
            scaled_state_dict[key] += scaled_update
        
        self.model.load_state_dict(scaled_state_dict)

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

