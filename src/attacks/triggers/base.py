from __future__ import annotations
from abc import ABC, abstractmethod
import torch

Tensor = torch.Tensor


class BaseTrigger(ABC):
    """
    Abstract class for backdoor triggers.

    This class provides a skeleton for different trigger implementations.
    Subclasses must implement the `apply` method.
    """

    def __init__(self, position, size, pattern, alpha=1.0):
        """
        Initializes the trigger's core attributes.
        Args:
            position (Tuple[int, int]): Top-left corner (x, y) of the trigger.
            size (Tuple[int, int]): Size (width, height) of the trigger.
            pattern: The trigger pattern itself (e.g., a color value, a small numpy array).
        """
        self.position = position
        self.size = size
        self.pattern = pattern
        self.alpha = alpha

    @abstractmethod
    def apply(self, image: Tensor) -> Tensor:
        """
        Applies the trigger to a given image.
        Args:
            image (Tensor): The input image tensor to which the trigger will be applied.
        Returns:
            Tensor: The modified image tensor with the trigger applied.
        """
        pass

