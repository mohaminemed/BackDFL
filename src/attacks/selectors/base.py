from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset

class BaseSelector(ABC):
    """
    Abstract base class for sample selectors.

    This class provides a blueprint for different strategies to select which
    samples from a dataset should be poisoned. Subclasses must implement
    the `select` method.
    """

    def __init__(self, poisoning_rate: float):
        """
        Initializes the selector with a poisoning rate.

        Args:
            poisoning_rate (float): The fraction of the dataset to select for
                                    poisoning (must be between 0.0 and 1.0).
        """
        if not 0.0 <= poisoning_rate <= 1.0:
            raise ValueError("Poisoning rate must be between 0.0 and 1.0.")
        self.poisoning_rate = poisoning_rate

    @abstractmethod
    def select(self, dataset: Dataset) -> List[int]:
        """
        Selects a subset of indices from the dataset to be poisoned.

        Args:
            dataset (Dataset): The dataset from which to select samples.

        Returns:
            List[int]: A list of integer indices of the samples to be poisoned.
        """
        pass