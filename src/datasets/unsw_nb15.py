import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .adapter import DatasetAdapter

_CATEGORICAL_COLS = ["proto", "service", "state"]
_DROP_COLS = ["id", "attack_cat", "label"]

_ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Backdoors", "Shellcode", "Worms",
]
# "Backdoor"/"Backdoors" naming differs across UNSW-NB15 CSV releases; merge them.
_CATEGORY_ALIASES = {"Backdoors": "Backdoor"}
_CATEGORY_TO_IDX = {
    "Normal": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3, "DoS": 4,
    "Reconnaissance": 5, "Analysis": 6, "Backdoor": 7, "Shellcode": 8, "Worms": 9,
}


class UNSWNB15Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class UNSWNB15Adapter(DatasetAdapter):
    """
    UNSW-NB15 network intrusion detection dataset.

    Expected directory layout:
        data/unsw_nb15/UNSW_NB15_training-set.csv
        data/unsw_nb15/UNSW_NB15_testing-set.csv

    Feature layout after preprocessing (fixed across train/test):
        [ standardized numeric features | one-hot proto | one-hot service | one-hot state ]

    Categorical vocabularies (incl. an "<UNK>" bucket for unseen values at test
    time) and numeric standardization stats are both fit on the training split
    only, then reused for the test split. Numeric features always occupy
    indices [0, num_numeric), which trigger implementations rely on to avoid
    corrupting the one-hot blocks.
    Target: 10-class attack category (normal + 9 attack families).
    """

    def __init__(self, root="data", train=True, download=False, seed=42):
        self.seed = seed
        super().__init__(root=root, train=train, download=download, transform=None)

    def load_dataset(self) -> None:
        base = os.path.join(self.root, "unsw_nb15")
        train_path = os.path.join(base, "UNSW_NB15_training-set.csv")
        test_path = os.path.join(base, "UNSW_NB15_testing-set.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        numeric_cols = [c for c in df_train.columns if c not in _CATEGORICAL_COLS + _DROP_COLS]
        self.num_numeric = len(numeric_cols)

        mean = df_train[numeric_cols].mean()
        std = df_train[numeric_cols].std().replace(0, 1.0)

        # Fit categorical vocab (+ "<UNK>") from train split only.
        vocabs = {col: sorted(df_train[col].astype(str).unique().tolist()) + ["<UNK>"] for col in _CATEGORICAL_COLS}

        df = df_train if self.train else df_test

        numeric = ((df[numeric_cols] - mean) / std).to_numpy(dtype=np.float32)

        onehot_blocks = []
        for col in _CATEGORICAL_COLS:
            vocab = vocabs[col]
            values = df[col].astype(str).where(df[col].astype(str).isin(vocab[:-1]), "<UNK>")
            codes = pd.Categorical(values, categories=vocab)
            onehot = pd.get_dummies(codes).reindex(columns=vocab, fill_value=0).to_numpy(dtype=np.float32)
            onehot_blocks.append(onehot)

        X = np.concatenate([numeric] + onehot_blocks, axis=1)

        attack_cat = df["attack_cat"].fillna("Normal").str.strip().replace(_CATEGORY_ALIASES)
        attack_cat = attack_cat.where(attack_cat.isin(_CATEGORY_TO_IDX), "Normal")
        y = attack_cat.map(_CATEGORY_TO_IDX).to_numpy(dtype=np.int64)

        self._dataset = UNSWNB15Dataset(X, y)

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_adapter = UNSWNB15Adapter(root=self.root, train=False, seed=self.seed)
        return DataLoader(test_adapter.dataset, batch_size=batch_size, shuffle=shuffle)
