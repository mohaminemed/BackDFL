import torch
from typing import Any, Dict, Optional
from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin


class GaussianAttackClient(AttackerAggregationMixin, BenignClient):
    """
    Malicious client performing Gaussian model poisoning.
    After finishing local training, the attacker replaces its entire
    model update with Gaussian noise drawn from N(0, variance).
    """

    def __init__(
        self,
        variance: float = 200.0,   # As specified in the paper
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = 0.0
        self.variance = float(variance)
        self.std = float(variance) ** 0.5

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Train normally (benign). Then poison the outgoing update.
        """
        # 1) Benign training
        result = super().local_train(epochs, round_idx)

        print(f"[GaussianAttackClient {self.get_id()}] Replacing update with Gaussian noise (var={self.variance}).")

        # 2) Poison outgoing model update
        gaussian_state = {}
        for k, v in self.model.state_dict().items():
            noise = torch.normal(
                mean=self.mean,
                std=self.std,
                size=v.shape,
                device=v.device,
                dtype=v.dtype,
            )
            gaussian_state[k] = noise

        # Overwrite model parameters with Gaussian noise
        self.model.load_state_dict(gaussian_state)

        # 3) Update the returned model weights so neighbors receive the poisoned version
        result["weights"] = self.get_params()
        return result
    
        
    

      # Return the state_dict attacker will expose
      #return self.model.state_dict()

    