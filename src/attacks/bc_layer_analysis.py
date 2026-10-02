import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BCLayerAnalyzer:
    """
    Implements Layer Substitution Analysis (Section 3 / Fig. 2 of
    "Backdoor Federated Learning by Poisoning Backdoor-Critical Layers", ICLR 2024)
    to identify the set L* of backdoor-critical (BC) layers for a malicious client.

    Pipeline:
      Step 1: train w_benign on clean data, then w_malicious by further training
              w_benign on poisoned data.
      Step 2: forward substitution b2m(l) - replace each layer l of w_malicious
              with the corresponding layer of w_benign, measure the BSR drop.
      Step 3: backward substitution m2b(L*) - incrementally copy layers from
              w_malicious into w_benign (most-critical first, per Step 2's
              ranking) until BSR reaches tau * BSR_malicious. The copied
              layers' indices define L*.

    Computational note: this is the O(n * l * t_forward) procedure the paper's
    own complexity analysis describes (Appendix 9) - one BSR evaluation pass
    per layer for both forward and backward substitution. It is intentionally
    run less frequently than every round in practice (see
    `bc_identification_interval` on the client classes; Fig. 6 of the paper
    shows reusing a stale L* for several rounds costs some BSR but not much).
    """

    def __init__(self, device: torch.device, tau: float = 0.95):
        self.device = device
        self.tau = tau

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)

    @staticmethod
    def _poisonable_param_names(model: nn.Module) -> List[str]:
        """Candidate layers for substitution: every trainable parameter tensor."""
        return [n for n, p in model.named_parameters() if p.requires_grad]

    def _train_on_loader(
        self, model: nn.Module, loader: DataLoader, epochs: int, lr: float, loss_fn: nn.Module
    ) -> nn.Module:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(data), target)
                loss.backward()
                opt.step()
        return model

    def compute_bsr(self, model: nn.Module, poison_val_loader: DataLoader, target_class: int) -> float:
        """Backdoor Success Rate: % of trigger-embedded validation samples classified as target_class."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, _ in poison_val_loader:
                data = data.to(self.device)
                pred = model(data).argmax(dim=1)
                correct += (pred == target_class).sum().item()
                total += data.size(0)
        return 100.0 * correct / max(total, 1)

    # ------------------------------------------------------------------ #
    # Step 1: Local Training
    # ------------------------------------------------------------------ #
    def train_benign_and_malicious(
        self,
        global_model: nn.Module,
        clean_train_loader: DataLoader,
        poison_train_loader: DataLoader,
        benign_epochs: int,
        malicious_epochs: int,
        lr: float,
        loss_fn: Optional[nn.Module] = None,
    ) -> Tuple[nn.Module, nn.Module]:
        loss_fn = loss_fn or nn.CrossEntropyLoss()

        w_benign = self._clone_model(global_model).to(self.device)
        w_benign = self._train_on_loader(w_benign, clean_train_loader, benign_epochs, lr, loss_fn)

        w_malicious = self._clone_model(w_benign).to(self.device)
        w_malicious = self._train_on_loader(w_malicious, poison_train_loader, malicious_epochs, lr, loss_fn)

        return w_benign, w_malicious

    # ------------------------------------------------------------------ #
    # Step 2: Forward Layer Substitution, b2m(l)
    # ------------------------------------------------------------------ #
    def forward_substitution(
        self,
        w_benign: nn.Module,
        w_malicious: nn.Module,
        poison_val_loader: DataLoader,
        target_class: int,
    ) -> Dict[str, float]:
        """Returns {layer_name: delta_BSR} where delta_BSR = BSR_malicious - BSR_b2m(l).
        A large positive value means removing that layer hurts the backdoor a lot
        -> the layer is a strong BC-layer candidate."""
        bsr_malicious = self.compute_bsr(w_malicious, poison_val_loader, target_class)
        layer_names = self._poisonable_param_names(w_malicious)
        benign_state = w_benign.state_dict()

        delta_bsr: Dict[str, float] = {}
        for name in layer_names:
            w_b2m = self._clone_model(w_malicious).to(self.device)
            tmp_state = w_b2m.state_dict()
            tmp_state[name] = benign_state[name].clone()
            w_b2m.load_state_dict(tmp_state)

            bsr_b2m = self.compute_bsr(w_b2m, poison_val_loader, target_class)
            delta_bsr[name] = bsr_malicious - bsr_b2m

        return delta_bsr

    # ------------------------------------------------------------------ #
    # Step 3: Backward Layer Substitution, m2b(L*)
    # ------------------------------------------------------------------ #
    def backward_substitution(
        self,
        w_benign: nn.Module,
        w_malicious: nn.Module,
        delta_bsr: Dict[str, float],
        poison_val_loader: DataLoader,
        target_class: int,
    ) -> List[str]:
        """Incrementally copies layers (highest delta_BSR first) from w_malicious
        into w_benign until BSR(w_m2b(L*)) >= tau * BSR_malicious. Returns L*,
        the ordered list of layer names copied (order = criticality, descending)."""
        bsr_malicious = self.compute_bsr(w_malicious, poison_val_loader, target_class)
        threshold = self.tau * bsr_malicious

        sorted_layers = sorted(delta_bsr.keys(), key=lambda n: delta_bsr[n], reverse=True)
        malicious_state = w_malicious.state_dict()

        l_star: List[str] = []
        w_m2b = self._clone_model(w_benign).to(self.device)

        for name in sorted_layers:
            l_star.append(name)
            tmp_state = w_m2b.state_dict()
            tmp_state[name] = malicious_state[name].clone()
            w_m2b.load_state_dict(tmp_state)

            bsr_m2b = self.compute_bsr(w_m2b, poison_val_loader, target_class)
            if bsr_m2b >= threshold:
                break

        return l_star

    # ------------------------------------------------------------------ #
    def identify_bc_layers(
        self,
        global_model: nn.Module,
        clean_train_loader: DataLoader,
        poison_train_loader: DataLoader,
        poison_val_loader: DataLoader,
        target_class: int,
        benign_epochs: int = 2,
        malicious_epochs: int = 5,
        lr: float = 0.01,
    ) -> Tuple[List[str], nn.Module, nn.Module]:
        """Full pipeline (Steps 1-3). Returns (L*, w_benign, w_malicious)."""
        w_benign, w_malicious = self.train_benign_and_malicious(
            global_model, clean_train_loader, poison_train_loader,
            benign_epochs, malicious_epochs, lr,
        )
        delta_bsr = self.forward_substitution(w_benign, w_malicious, poison_val_loader, target_class)
        l_star = self.backward_substitution(w_benign, w_malicious, delta_bsr, poison_val_loader, target_class)
        return l_star, w_benign, w_malicious