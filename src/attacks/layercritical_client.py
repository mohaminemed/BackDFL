from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from ..fl.baseclient import BenignClient
from .aggregation.mixing import AttackerAggregationMixin
from ..datasets.backdoor import BackdoorDataset
from .selectors.base import BaseSelector
from .bc_layer_analysis import BCLayerAnalyzer
from .triggers.patch import PatchTrigger  # existing fixed-pattern trigger (BaseTrigger subclass)


class LPClient(AttackerAggregationMixin, BenignClient):
    """
    Layer-wise Poisoning (LP) attack (Section 4.1 of "Backdoor Federated
    Learning by Poisoning Backdoor-Critical Layers", ICLR 2024).

    Identifies backdoor-critical (BC) layers L* via Layer Substitution
    Analysis, then crafts a malicious update that strongly poisons only
    those layers while pulling every other layer toward an estimate of the
    honest average model (Eq. 1-4). This minimizes the crafted update's
    distance from benign updates, bypassing distance-based defenses
    (MultiKrum, FLAME, FLTrust) with far less model deviation than
    whole-model attacks like DBA or Model Replacement.
    """

    def __init__(
        self,
        selector: BaseSelector,
        target_class: int,
        trigger: PatchTrigger,  # e.g. PatchTrigger(position=(28,28), size=(5,5), color=(1,1,1))
                                 # for a BadNets-style 5x5 square, per the paper's baseline
        attack_start_round: int = 1,
        attack_end_round: int = -1,
        poison_rate: float = 0.25,          # paper default PDR = 50% (Table A-7)
        tau: float = 0.95,                     # BC layer identification threshold
        lam: float = 1.0,                      # stealthiness knob, Eq. 4 (paper: 1.0 on CIFAR, 0.5 on FMNIST)
        bc_identification_interval: int = 1,   # re-run Layer Substitution Analysis every N attack rounds
        benign_epochs: int = 5,
        malicious_epochs: int = 5,
        local_lr: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.selector = selector
        self.target_class = int(target_class)
        self.trigger = trigger
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round if attack_end_round > 0 else float('inf')
        self.poison_fraction = poison_rate
        self.tau = tau
        self.lam = lam
        self.bc_identification_interval = bc_identification_interval
        self.benign_epochs = benign_epochs
        self.malicious_epochs = malicious_epochs
        self.local_lr = local_lr

        self.analyzer = BCLayerAnalyzer(device=self.device, tau=self.tau)
        self._cached_l_star: Optional[List[str]] = None
        self._cached_w_benign_state: Optional[Dict[str, torch.Tensor]] = None
        self._cached_w_malicious_state: Optional[Dict[str, torch.Tensor]] = None
        self._rounds_since_identification = 0

        # Peer malicious clients' locally-trained benign models, if the
        # harness wires up inter-malicious-client communication (the paper's
        # threat model, Sec 2.2, assumes malicious clients can coordinate).
        # Falls back to using only this client's own benign model as the
        # u_average estimate if no peers are registered.
        self.peer_benign_states: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------ #
    def _split_clean_poison(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Splits local data into D_clean and D_poison (Sec 3.2)."""
        base_dataset = self.trainloader.dataset
        n = len(base_dataset)
        n_poison = max(1, int(n * self.poison_fraction))
        n_clean = max(1, n - n_poison)
        clean_ds, poison_base_ds = random_split(base_dataset, [n_clean, n_poison])

        poison_ds = BackdoorDataset(
            original_dataset=poison_base_ds,
            trigger_fn=self.trigger.apply,
            target_label=self.target_class,
            poison_fraction=1.0,  # every sample reaching this loader is poisoned
            seed=42,
            poison_exclude_target=True,
        )

        batch_size = getattr(self.trainloader, "batch_size", 32)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=True)
        poison_loader = DataLoader(poison_ds, batch_size=batch_size, shuffle=True)

        # Held-out poisoned validation split (D_poison,val) used only for BSR
        # evaluation during Layer Substitution Analysis.
        val_size = max(1, n_poison // 5)
        poison_val_ds = Subset(poison_ds, list(range(min(val_size, len(poison_ds)))))
        poison_val_loader = DataLoader(poison_val_ds, batch_size=batch_size, shuffle=False)

        return clean_loader, poison_loader, poison_val_loader

    def _identify_or_refresh_bc_layers(
        self, clean_loader: DataLoader, poison_loader: DataLoader, poison_val_loader: DataLoader, round_idx: int
    ) -> None:
        """Runs full Layer Substitution Analysis, or reuses the cached L* and
        just refreshes w_benign/w_malicious by brief local retraining
        (Fig. 6: reusing a stale L* for a few rounds costs some BSR but
        remains effective - identification doesn't need to run every round)."""
        need_identification = (
            self._cached_l_star is None
            or self._rounds_since_identification >= self.bc_identification_interval
        )
        if need_identification:
            print(f"\n--- LP/LF Client [{self.get_id()}] running Layer Substitution Analysis (round {round_idx}) ---")
            l_star, w_benign, w_malicious = self.analyzer.identify_bc_layers(
                global_model=self.model,
                clean_train_loader=clean_loader,
                poison_train_loader=poison_loader,
                poison_val_loader=poison_val_loader,
                target_class=self.target_class,
                benign_epochs=self.benign_epochs,
                malicious_epochs=self.malicious_epochs,
                lr=self.local_lr,
            )
            self._cached_l_star = l_star
            self._rounds_since_identification = 0
            print(f"[Client {self.get_id()}] Identified {len(l_star)} BC layers: {l_star}")
        else:
            w_benign, w_malicious = self.analyzer.train_benign_and_malicious(
                global_model=self.model,
                clean_train_loader=clean_loader,
                poison_train_loader=poison_loader,
                benign_epochs=self.benign_epochs,
                malicious_epochs=self.malicious_epochs,
                lr=self.local_lr,
            )
            self._rounds_since_identification += 1

        self._cached_w_benign_state = w_benign.state_dict()
        self._cached_w_malicious_state = w_malicious.state_dict()

    def _estimate_average_benign_model(self, local_benign_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """u_average = mean of local benign models across malicious clients (Eq. 4)."""
        states = [local_benign_state] + self.peer_benign_states
        return {k: torch.stack([s[k].float() for s in states], dim=0).mean(dim=0) for k in local_benign_state.keys()}

    def _craft_lp_update(
        self,
        l_star: List[str],
        w_malicious_state: Dict[str, torch.Tensor],
        u_average_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Eq. 4: w~^(i) = lam*v*u_malicious + ReLU(1-lam)*v*u_average + (1-v)*u_average
        v = 1 on L* layers, 0 elsewhere.
        """
        crafted = {}
        relu_term = max(1.0 - self.lam, 0.0)
        l_star_set = set(l_star)
        for name, u_avg in u_average_state.items():
            if name in l_star_set:
                u_mal = w_malicious_state[name].to(u_avg.dtype)
                crafted[name] = self.lam * u_mal + relu_term * u_avg
            else:
                crafted[name] = u_avg.clone()
        return crafted

    def _adaptive_layer_control(
        self,
        l_star: List[str],
        w_malicious_state: Dict[str, torch.Tensor],
        u_average_state: Dict[str, torch.Tensor],
        simulate_rejection_fn=None,
    ) -> List[str]:
        """
        Adaptive layer control (Sec 4.1): if the crafted model would be
        rejected by the server's distance-based defense, shrink L* by
        dropping the least-critical layer (the one added last during
        backward substitution, i.e. the tail of l_star) and retry.

        `simulate_rejection_fn(crafted_state) -> bool` should implement the
        harness's own simulated MultiKrum/FLAME acceptance test using the
        attacker's locally available benign models as a stand-in for the
        true benign-client population, as described in the paper. If not
        provided, no shrinking is performed (full L* is used) - wire up a
        real simulated-defense callback for a faithful reproduction of the
        paper's adaptive behavior.
        """
        if simulate_rejection_fn is None:
            return list(l_star)

        candidate = list(l_star)
        while candidate:
            crafted = self._craft_lp_update(candidate, w_malicious_state, u_average_state)
            if not simulate_rejection_fn(crafted):
                break
            if len(candidate) <= 1:
                break
            candidate.pop()  # drop the least-critical layer (end of l_star's descending-criticality order)
        return candidate

    # ------------------------------------------------------------------ #
    def local_train(
        self, epochs: int, round_idx: int, simulate_rejection_fn=None, **kwargs
    ) -> Dict[str, Any]:
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            print(f"Client [{self.get_id()}]: Behaving benignly in round {round_idx} (outside attack window).")
            return super().local_train(epochs, round_idx)

        clean_loader, poison_loader, poison_val_loader = self._split_clean_poison()
        self._identify_or_refresh_bc_layers(clean_loader, poison_loader, poison_val_loader, round_idx)

        u_average_state = self._estimate_average_benign_model(self._cached_w_benign_state)

        effective_l_star = self._adaptive_layer_control(
            self._cached_l_star, self._cached_w_malicious_state, u_average_state, simulate_rejection_fn,
        )

        crafted_state = self._craft_lp_update(
            effective_l_star, self._cached_w_malicious_state, u_average_state,
        )

        target_state = self.model.state_dict()
        for k, v in crafted_state.items():
            target_state[k] = v.to(target_state[k].device).to(target_state[k].dtype)
        self.model.load_state_dict(target_state)

        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
            'round_idx': round_idx,
            'bc_layers': effective_l_star,
        }


class LFClient(LPClient):
    """
    Layer-wise Flipping (LF) attack (Section 4.2): targets sign-based
    defenses such as RLR, which detect per-parameter sign disagreement
    between a client's update and the majority, then flip the learning
    rate's sign for those parameters to neutralize the update.

    LF proactively flips the sign of the BC-layer update:
        w_LFA^(i) := -(w_m2b(L*) - w) + w  =  2w - w_m2b(L*)
    so that when the server's sign-based defense detects the "wrong" sign
    and flips the learning rate back, the correction restores the actual
    malicious direction instead of neutralizing it.

    Note (paper Sec 5.3): LF is ineffective against defenses that don't do
    sign-based correction, and the paper reports it fails on large models
    where RLR itself fails to reliably reverse signs.
    """

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            print(f"Client [{self.get_id()}]: Behaving benignly in round {round_idx} (outside attack window).")
            return BenignClient.local_train(self, epochs, round_idx)

        clean_loader, poison_loader, poison_val_loader = self._split_clean_poison()
        self._identify_or_refresh_bc_layers(clean_loader, poison_loader, poison_val_loader, round_idx)

        l_star_set = set(self._cached_l_star)
        global_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # w_m2b(L*): global model with BC layers replaced by the malicious model's layers.
        w_m2b_state = {k: v.clone() for k, v in global_state.items()}
        for name in l_star_set:
            w_m2b_state[name] = self._cached_w_malicious_state[name].clone()

        # w_LFA = -(w_m2b(L*) - w) + w = 2w - w_m2b(L*)
        flipped_state = {
            k: (2.0 * global_state[k].float() - w_m2b_state[k].float()).to(global_state[k].dtype)
            for k in global_state.keys()
        }

        self.model.load_state_dict(flipped_state)

        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
            'round_idx': round_idx,
            'bc_layers': list(l_star_set),
        }