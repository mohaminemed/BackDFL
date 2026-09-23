import torch
from .base import BaseTrigger


class NBaIoTTrigger(BaseTrigger):
    """
    Backdoor trigger for N-BaIoT (tabular, all-numeric standardized features).

    Unlike NSL-KDD/UNSW-NB15 there is no one-hot block here, but the 115
    features are stream statistics computed over several correlated
    time-decay windows (L5/L3/L1/L0.1/L0.01) of the same underlying signal.
    Perturbing many of them independently (or windows that should track each
    other) produces a statistically implausible traffic fingerprint, which is
    both unrealistic and easy to flag. This trigger therefore only perturbs a
    small, fixed number of *independent, longest-window* feature indices
    (default: first 3, i.e. the coarse L5 host-stats), leaving the shorter,
    strongly-correlated windows untouched.
    """

    def __init__(self, trigger_features=(0, 1, 2), trigger_value=3.0):
        self.trigger_features = trigger_features
        self.trigger_value = trigger_value

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for f in self.trigger_features:
            x[f] += self.trigger_value
        return x

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.clone()
        batch[:, list(self.trigger_features)] += self.trigger_value
        return batch
