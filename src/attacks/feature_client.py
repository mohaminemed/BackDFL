from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin

class FeatureAttackClient(AttackerAggregationMixin, BenignClient):
    """
    A malicious client performing a feature-level poisoning attack.
    It replaces features of a fraction of local training examples
    with Gaussian noise (mean=0, variance=1000).
    """

    def __init__(
        self,
        poison_fraction: float = 1.0,  # fraction of local data to poison
        mean: float = 0.0,
        variance: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.poison_fraction = poison_fraction
        self.mean = mean
        self.variance = variance

        # Apply feature poisoning to the local dataset
        self.poison_local_data()

    def poison_local_data(self):
        """
        Replace features of a fraction of local dataset examples with Gaussian noise.
        """
        dataset = self.trainloader.dataset
        poisoned_data, poisoned_labels = [], []

        for idx in range(len(dataset)):
            x, y = dataset[idx]
            if torch.rand(1).item() < self.poison_fraction:
                noise = torch.normal(mean=self.mean, std=self.variance**0.5, size=x.shape)
                x = noise.to(x.device)
            poisoned_data.append(x)
            poisoned_labels.append(y)

        poisoned_dataset = TensorDataset(torch.stack(poisoned_data), torch.stack(poisoned_labels))
        self.trainloader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Train normally on the poisoned dataset.
        """
        self._model.train()
        for _ in range(epochs):
            for inputs, targets in self.trainloader:
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

    