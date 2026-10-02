from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin

class KrumAttackClient(AttackerAggregationMixin, BenignClient):
    """
    Krum attack that crafts local model parameters after training,
    before sending them to neighbors.
    """

    def __init__(self, epsilon: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Train normally, THEN craft the malicious update before returning it.
        """
        # ---- Train normally (benign) ----
        out = super().local_train(epochs, round_idx)

        # ----------------------------------------------------
        # At this moment, training is finished and it's time to poison the update.
        # ----------------------------------------------------

        # The attacker needs neighbor honest updates to compute centroid...
        # But we do NOT have them here, because neighbors send AFTER us.
        #
        # So for decentralized FL, the standard way to do Krum attack is:
        #
        # → Craft a malicious update that is *extremely small norm* 
        #   so that Krum always picks it (since it is closest to all others).
        #
        # This is the decentralized-compatible Krum attack variant.
        # ----------------------------------------------------

        print(f"[KrumAttackClient {self.get_id()}] Crafting low-norm malicious update.")

        poisoned_state = {}
        current_state = self.model.state_dict()

        for k, v in current_state.items():
            # small-norm vector that makes attacker appear closest to everyone
            poisoned_state[k] = torch.zeros_like(v) + self.epsilon * torch.randn_like(v)

        # load the malicious update into the model so DFL will broadcast it
        self.model.load_state_dict(poisoned_state)

        # update the returned parameters
        out["weights"] = self.get_params()

        return out
    
        
    

      # Return the state_dict attacker will expose
      #return self.model.state_dict()

    