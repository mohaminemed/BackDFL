import torch
from typing import Dict, Any, Optional
from copy import deepcopy

from ..fl.baseclient import BenignClient


class OMPClient(BenignClient):
    """
    Implementation of NDSS'21 manipulating the byzantine: optimized model poisoning attack.

    Core idea:
    - Estimate benign update
    - Compute adversarial direction
    - Scale with gamma
    - Replace local update
    """

    def __init__(self, 
       gamma_init: float = 1.0, 
       tau: float = 1e-3,
       perturbation: str = "sign",  # "sign" or "unit"
       malicious_epochs: int = 1,
       attack_start_round: int = 1,
       attack_end_round: int = -1,
       **kwargs):
        super().__init__(**kwargs)
        

        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round

        self.gamma_init = gamma_init
        self.tau = tau

        self.perturbation = perturbation  # sign | unit
        self.malicious_epochs = malicious_epochs  # 

    # ---------------------------------------------------------
    # STEP 1: Compute benign update
    # ---------------------------------------------------------
    def _compute_benign_update(self):
        original_weights = deepcopy(self.get_params())

        super().local_train(epochs=self.malicious_epochs, round_idx=0)

        new_weights = self.get_params()

        delta = {}
        for k in original_weights:
            delta[k] = new_weights[k] - original_weights[k]

        return delta

    # ---------------------------------------------------------
    # STEP 2: Compute perturbation direction
    # ---------------------------------------------------------
    def _compute_direction(self, benign_update):
        direction = {}

        for k, v in benign_update.items():
            if self.perturbation == "sign":
                direction[k] = -torch.sign(v)
            elif self.perturbation == "unit":
                norm = torch.norm(v) + 1e-12
                direction[k] = -v / norm
            else:
                direction[k] = -torch.sign(v)

        return direction

    # ---------------------------------------------------------
    # STEP 3: Apply perturbation
    # ---------------------------------------------------------
    def _apply_attack(self, benign_update, direction, gamma):
        malicious_update = {}

        for k in benign_update:
            malicious_update[k] = benign_update[k] + gamma * direction[k]

        return malicious_update

    # ---------------------------------------------------------
    # STEP 4: Gamma search (Algorithm 1)
    # ---------------------------------------------------------
    def _optimize_gamma(self, benign_update, direction):
        gamma = self.gamma_init
        step = gamma / 2

        best_gamma = gamma

        for _ in range(10):  # iterative search
            test_update = self._apply_attack(benign_update, direction, gamma)

            if self._is_stealthy(test_update, benign_update):
                best_gamma = gamma
                gamma += step
            else:
                gamma -= step

            step /= 2

        return best_gamma

    # ---------------------------------------------------------
    # STEALTH constraint (approximation of Min-Max)
    # ---------------------------------------------------------
    def _is_stealthy(self, malicious_update, benign_update):
        # Simple proxy: bound L2 norm
        total_norm = 0.0
        benign_norm = 0.0

        for k in malicious_update:
            total_norm += torch.norm(malicious_update[k])**2
            benign_norm += torch.norm(benign_update[k])**2

        return total_norm <= 1.2 * benign_norm  # heuristic

    # ---------------------------------------------------------
    # MAIN
    # ---------------------------------------------------------
    def local_train(self, round_idx: int, epochs: int = 1, **kwargs):

        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)

        print(f"\n--- NDSS Poisoning Client [{self.id}] attacking round {round_idx} ---")

        # Step 1: benign update
        benign_update = self._compute_benign_update()

        # Step 2: direction
        direction = self._compute_direction(benign_update)

        # Step 3: gamma optimization
        gamma = self._optimize_gamma(benign_update, direction)

        # Step 4: malicious update
        malicious_update = self._apply_attack(benign_update, direction, gamma)

        # Step 5: apply to model
        new_weights = self.get_params()
        for k in new_weights:
            new_weights[k] += malicious_update[k]

        self.set_params(new_weights)

        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {'gamma': gamma},
            'round_idx': round_idx
        }