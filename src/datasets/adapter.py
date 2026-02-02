# src/datasets/adapter.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torch

class DatasetAdapter(ABC):
    """
    Abstract adapter for datasets in FL experiments.
    Concrete subclasses must implement `load_dataset()` to populate self._dataset.
    """

    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self._dataset: Optional[Dataset] = None
        self.load_dataset()

    @abstractmethod
    def load_dataset(self) -> None:
        """Load a torchvision-like dataset into self._dataset."""
        raise NotImplementedError

    @property
    def dataset(self) -> Dataset:
        assert self._dataset is not None, "Dataset not loaded"
        return self._dataset

    def get_test_loader(self, batch_size: int = 256, shuffle: bool = False) -> DataLoader:
        """Return a DataLoader for evaluation (test/val). For adapters that only load train/test separately,
        call with train=False when constructing adapter instance for test dataset."""
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def get_client_loaders(
        self,
        num_clients: int,
        strategy: str = "iid",
        batch_size: int = 64,
        seed: int = 0,
        **strategy_args
    ) -> Dict[int, DataLoader]:
        """
        Partition the underlying dataset and return a dict of DataLoaders (one per client).
        strategy: 'iid' or 'dirichlet'
        strategy_args: passed to partition helpers (e.g., alpha for dirichlet)
        """
        N = len(self.dataset)
        # extract labels (common attr names)
        labels = self._extract_labels(self.dataset)
        if strategy.lower() == "iid":
            parts = self.partition_iid(N, num_clients, seed=seed)
        elif strategy.lower() == "dirichlet":
            alpha = float(strategy_args.get("alpha", 0.5))
            parts = self.partition_dirichlet(labels, num_clients, alpha=alpha, seed=seed)
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

        loaders: Dict[int, DataLoader] = {}
        for cid, idxs in parts.items():
            subset = Subset(self.dataset, idxs)
            loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)
        return loaders

    # ---------- partition helpers ----------
    @staticmethod
    def partition_iid(N: int, num_clients: int, seed: int = 0) -> Dict[int, List[int]]:
        rng = np.random.RandomState(seed)
        indices = np.arange(N)
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return {i: list(map(int, s)) for i, s in enumerate(splits)}

    @staticmethod
    def partition_dirichlet(labels: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 0) -> Dict[int, List[int]]:
        """
        Dirichlet non-iid partition (per-class sampling).
        labels: np.array of length N with integer class labels.
        """
        rng = np.random.RandomState(seed)
        labels = np.asarray(labels)
        N = len(labels)
        num_classes = int(labels.max()) + 1
        indices = np.arange(N)
        client_indices = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            c_idxs = indices[labels == c]
            if len(c_idxs) == 0:
                continue
            rng.shuffle(c_idxs)
            # sample proportions for this class across clients
            proportions = rng.dirichlet([alpha] * num_clients)
            counts = (proportions * len(c_idxs)).astype(int)

            # fix rounding shortfall
            shortfall = len(c_idxs) - counts.sum()
            if shortfall > 0:
                order = np.argsort(proportions)[::-1]
                for j in range(shortfall):
                    counts[order[j % len(order)]] += 1

            start = 0
            for i in range(num_clients):
                cnt = counts[i]
                if cnt > 0:
                    client_indices[i].extend(list(c_idxs[start:start + cnt]))
                    start += cnt

        # shuffle each client's indices
        for i in client_indices:
            rng.shuffle(client_indices[i])

        return {i: list(map(int, client_indices[i])) for i in client_indices}

    # ---------- small helpers ----------
    @staticmethod
    def _extract_labels(ds: Dataset) -> np.ndarray:
        """Try common dataset attributes (.targets, .labels, .samples) to extract labels."""
        if hasattr(ds, "targets"):
            return np.asarray(ds.targets)
        if hasattr(ds, "labels"):
            return np.asarray(ds.labels)
        if hasattr(ds, "samples"):
            return np.asarray([s[1] for s in ds.samples])
        # fallback: iterate (may be slow)
        labels = []
        for i in range(len(ds)):
            _, y = ds[i]
            labels.append(int(y))
        return np.asarray(labels)
