import torch
from typing import Any, Dict, Optional
from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin


class TrimAttackClient(AttackerAggregationMixin, BenignClient):
    """
    Malicious client performing a Trim attack.
    After training, the client's update is manipulated to maximize
    deviation from the benign aggregation.
    """

    def __init__(
        self,
        scale: float = 5.0,  # Magnitude of the deviation from benign update
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Train normally and then poison the outgoing model by scaling the delta
        from the initial parameters.
        """
        # 1) Store original model state before training
        original_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # 2) Perform standard benign training
        result = super().local_train(epochs, round_idx)

        # 3) Manipulate the update: scale the delta from the original model
        manipulated_state = {}
        for k, v in self.model.state_dict().items():
            delta = v - original_state[k]
            manipulated_state[k] = original_state[k] + self.scale * delta

        # 4) Load manipulated state into the model
        self.model.load_state_dict(manipulated_state)

        # 5) Update the returned weights so neighbors receive the poisoned version
        result["weights"] = self.get_params()

        print(f"[TrimAttackClient {self.get_id()}] Applied Trim attack with scale={self.scale}.")
        return result

        
    

      # Return the state_dict attacker will expose
      #return self.model.state_dict()

    