import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .adapter import DatasetAdapter
from typing import Optional


class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HARAdapter(DatasetAdapter):
    """
    Human Activity Recognition (UCI HAR)

    - 30 users
    - 6 classes
    - 561 features
    - 75% train / 25% test per user
    """

    def __init__(
        self,
        root="data/har",
        train=True,
        download=False,
        seed=42,
    ):
        self.seed = seed
        super().__init__(root=root, train=train, download=download, transform=None)

    def load_dataset(self) -> None:
        X, y, subjects = self._load_raw()

        rng = np.random.RandomState(self.seed)

        train_idx = []
        test_idx = []

        # Subject-wise 75/25 split (as in [8])
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            rng.shuffle(idx)

            split = int(0.75 * len(idx))
            train_idx.extend(idx[:split])
            test_idx.extend(idx[split:])

        if self.train:
            indices = train_idx
        else:
            indices = test_idx

        self._dataset = HARDataset(X[indices], y[indices])

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_adapter = HARAdapter(
            root=self.root,
            train=False,
            seed=self.seed,
        )

        return DataLoader(test_adapter.dataset, batch_size=batch_size, shuffle=shuffle)

    # --------------------------------------------------

    def _load_raw(self):
        """
        Expected directory:

        data/har/
            ├── features.txt
            ├── train/
            └── test/

        We merge official train+test then redo subject-wise split.
        """

        base = os.path.join(self.root, "har")

        def load_split(split):
            X = np.loadtxt(os.path.join(base, split, f"X_{split}.txt"))
            y = np.loadtxt(os.path.join(base, split, f"y_{split}.txt")) - 1  # zero-based
            s = np.loadtxt(os.path.join(base, split, f"subject_{split}.txt"))
            return X, y.astype(int), s.astype(int)

        Xtr, ytr, str_ = load_split("train")
        Xte, yte, ste = load_split("test")

        X = np.vstack([Xtr, Xte])
        y = np.concatenate([ytr, yte])
        subjects = np.concatenate([str_, ste])

        return X, y, subjects
