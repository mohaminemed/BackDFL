import torch
from .base import BaseTrigger


class HARTrigger(BaseTrigger):
    """
    Backdoor trigger for HAR (tabular 561-dim vectors)
    """

    def __init__(self, trigger_features=(0, 10, 20), trigger_value=5.0):
        self.trigger_features = trigger_features
        self.trigger_value = trigger_value

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [561]

        x = x.clone()

        for f in self.trigger_features:
            x[f] += self.trigger_value

        return x
