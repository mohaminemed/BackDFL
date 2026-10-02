from typing import Any, Dict, Optional
import torch

from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .triggers.distributed import DBATrigger


class DBAClient(AttackerAggregationMixin, BenignClient):
    """
    A malicious client for the Distributed Backdoor Attack (DBA).

    This client's logic is identical to a BadNets client; it trains naively
    on a poisoned dataset. The distinction is that it is initialized with a
    DBATrigger, which only applies a small part of a global trigger pattern.
    """

    def __init__(
        self,
        selector: BaseSelector,
        trigger: DBATrigger,
        target_class: int,
        attack_start_round: int,
        attack_end_round: int,
        poison_fraction: float = 0.1,
        malicious_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(trigger, DBATrigger):
            raise TypeError("DBAClient must be initialized with a DBATrigger instance.")
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class
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

        self.poisoned_dataloader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True
          )
      
        num_poisoned = sum(1 for _ in self.poisoned_dataloader)  
        print(f"[DBAClient {self.get_id()}] poisoned loader created with ~{num_poisoned} batches, target_class={self.target_class}")

        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round
        self.poison_fraction = poison_fraction

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the benign training process to perform a DBA attack.
        """
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        
        
        # Perform standard training on the poisoned dataloader
        train_loss, correct, total = 0.0, 0, 0
        
        for _ in range(self.malicious_epochs or epochs):
            if self.poisoned_dataloader is None: break
            for inputs, targets in self.poisoned_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        avg_loss = train_loss / (num_batches * (self.malicious_epochs or epochs))
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        return result


      # Return the state_dict attacker will expose
      #return self.model.state_dict()


