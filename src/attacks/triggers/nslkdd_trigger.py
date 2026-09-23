import torch
from .base import BaseTrigger


class NSLKDDTrigger(BaseTrigger):
    """
    Backdoor trigger for NSL-KDD (tabular, standardized numeric features +
    one-hot categorical blocks).

    IMPORTANT: NSL-KDD feature vectors mix standardized continuous features
    (indices [0, num_numeric)) with one-hot encoded categorical blocks
    (protocol_type/service/flag) appended afterwards. Perturbing a one-hot
    block directly (e.g. adding a constant) can produce an invalid encoding
    (multiple/no "hot" entries), which is both unrealistic for a real network
    flow and trivially detectable. This trigger therefore only perturbs a
    small, fixed subset of the *numeric* feature indices, mimicking a subtle
    shift in traffic statistics (e.g. counts/rates) rather than corrupting
    protocol identity.
    """

    def __init__(self, trigger_features=(0, 4, 5, 22, 23), trigger_value=3.0, num_numeric=38):
        if any(f >= num_numeric for f in trigger_features):
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
