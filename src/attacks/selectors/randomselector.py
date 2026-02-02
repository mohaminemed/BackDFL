import random
from typing import List
from torch.utils.data import Dataset
from .base import BaseSelector

class RandomSelector(BaseSelector):
    """
    Selects a random subset of samples from a dataset to be poisoned.
    """

    def __init__(self, poisoning_rate: float = 0.1):
        """
        Initializes the random selector.

        Args:
            poisoning_rate (float): The fraction of the dataset to randomly
                                    select (e.g., 0.1 for 10%).
        """
        super().__init__(poisoning_rate)

    def select(self, dataset: Dataset) -> List[int]:
        """
        Randomly selects indices from the dataset without replacement.

        Args:
            dataset (Dataset): The dataset from which to select samples.

        Returns:
            List[int]: A list of randomly chosen indices.
        """
        num_samples = len(dataset)
        num_to_poison = int(num_samples * self.poisoning_rate)

        # Generate a list of all possible indices
        all_indices = list(range(num_samples))
        
        # Sample a unique subset of indices
        poisoned_indices = random.sample(all_indices, num_to_poison)

        return poisoned_indices