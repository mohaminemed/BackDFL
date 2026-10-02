from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin

class LabelFlipClient(AttackerAggregationMixin, BenignClient):
    """
    A malicious client performing untargeted Label Flipping (LF) attack.
    It flips class labels of its local training dataset according to the LF strategy.
    """

    def __init__(
        self,
        flip_map: Dict[int, int],  # mapping old_label -> new_label
        poison_fraction: float = 1.0,  # fraction of local data to poison
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flip_map = flip_map
        self.poison_fraction = poison_fraction

        # Apply label flipping to a fraction of the local dataset
        self.poison_local_data()

    def poison_local_data(self):
        """
        Modify a fraction of the local training dataset labels according to flip_map.
        """
        dataset = self.trainloader.dataset

        # Support for datasets that are TensorDataset or custom dataset with __getitem__
        poisoned_data, poisoned_labels = [], []

        for idx in range(len(dataset)):
            x, y = dataset[idx]
            if torch.rand(1).item() < self.poison_fraction and y.item() in self.flip_map:
                y = torch.tensor(self.flip_map[y.item()], dtype=y.dtype)
            poisoned_data.append(x)
            poisoned_labels.append(y)

        # Create a new DataLoader with poisoned data
        poisoned_dataset = TensorDataset(torch.stack(poisoned_data), torch.stack(poisoned_labels))
        self.trainloader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Train normally but on the poisoned dataset.
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

    