import torch
from .base import BaseTrigger


class UNSWTrigger(BaseTrigger):
    """
    Backdoor trigger for UNSW-NB15 (tabular, standardized numeric features +
    one-hot categorical blocks).

    Same rationale as NSLKDDTrigger: the feature vector layout is
    [numeric features (indices [0, num_numeric))] followed by one-hot blocks
    for proto/service/state. Perturbing the one-hot tail would break the
    categorical encoding (yielding an invalid/ambiguous protocol) and make
    poisoned samples trivially distinguishable, so only numeric indices are
    ever perturbed here.
    """

    def __init__(self, trigger_features=(0, 1, 2, 3), trigger_value=3.0, num_numeric=None):
        if num_numeric is not None and any(f >= num_numeric for f in trigger_features):
            raise ValueError("trigger_features must index into the numeric feature block only")
        self.trigger_features = trigger_features
        self.trigger_value = trigger_value
        self.num_numeric = num_numeric

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for f in self.trigger_features:
            x[f] += self.trigger_value
        return x

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.clone()
        batch[:, list(self.trigger_features)] += self.trigger_value
        return batch
