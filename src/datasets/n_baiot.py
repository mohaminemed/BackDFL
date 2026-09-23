import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .adapter import DatasetAdapter

# 9 commercial IoT devices used in the original N-BaIoT study.
DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

# Only benign + gafgyt (BASHLITE) traffic is used: unlike mirai_attacks (shipped
# as per-device .rar archives), gafgyt_attacks/*.csv ships as plain CSV for all
# 9 devices, so no external archive tool is required to build this dataset.
_GAFGYT_SUBTYPES = ["combo", "junk", "scan", "tcp", "udp"]
_CATEGORY_TO_IDX = {"benign": 0, "combo": 1, "junk": 2, "scan": 3, "tcp": 4, "udp": 5}


class NBaIoTDataset(Dataset):
    def __init__(self, X, y, device_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.device_ids = np.asarray(device_ids)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NBaIoTAdapter(DatasetAdapter):
    """
    N-BaIoT: real traffic statistics (115 numeric features, no categorical
    columns) captured from 9 commercial IoT devices, benign vs BASHLITE
    (gafgyt) botnet traffic.

    Expected directory layout (official UCI release, per device):
        data/n_baiot/<device>/benign_traffic.csv
        data/n_baiot/<device>/gafgyt_attacks/{combo,junk,scan,tcp,udp}.csv

    Target: 6-class (benign + 5 gafgyt attack subtypes).

    Unlike NSL-KDD/UNSW-NB15, all 115 features are numeric statistical
    aggregates (packet-stream mean/std/... over several decay windows), so
    there is no one-hot block to avoid when designing a trigger — but the
    features are correlated across time-scales (L5/L3/L1/L0.1/L0.01), so
    triggers should still only touch a couple of independent long-window
    features rather than the whole vector, to keep poisoned samples close to
    plausible traffic statistics (see NBaIoTTrigger).

    Supports device-based client partitioning (`strategy="device"` in
    get_client_loaders), which maps each client to (a shard of) one physical
    IoT device — the natural, non-synthetic non-IID split for this dataset.
    """

    def __init__(self, root="data", train=True, download=False, seed=42, test_fraction=0.25):
        self.seed = seed
        self.test_fraction = test_fraction
        super().__init__(root=root, train=train, download=download, transform=None)

    def load_dataset(self) -> None:
        base = os.path.join(self.root, "n_baiot")

        rows_X, rows_y, rows_dev = [], [], []
        rows_X_train = []  # train-only rows, always collected, used to fit standardization stats
        rng = np.random.RandomState(self.seed)

        for device in DEVICES:
            device_dir = os.path.join(base, device)
            if not os.path.isdir(device_dir):
                continue

            frames = []
            benign_path = os.path.join(device_dir, "benign_traffic.csv")
            if os.path.isfile(benign_path):
                df = pd.read_csv(benign_path)
                df["__label"] = "benign"
                frames.append(df)

            for subtype in _GAFGYT_SUBTYPES:
                path = os.path.join(device_dir, "gafgyt_attacks", f"{subtype}.csv")
                if os.path.isfile(path):
                    df = pd.read_csv(path)
                    df["__label"] = subtype
                    frames.append(df)

            if not frames:
                continue

            df_dev = pd.concat(frames, ignore_index=True)
            feature_cols = [c for c in df_dev.columns if c != "__label"]
            if not hasattr(self, "feature_cols"):
                self.feature_cols = feature_cols

            # Deterministic per-device train/test split (no official split for N-BaIoT).
            idx = np.arange(len(df_dev))
            rng.shuffle(idx)
            split = int((1 - self.test_fraction) * len(idx))
            train_idx, test_idx = idx[:split], idx[split:]
            split_idx = train_idx if self.train else test_idx

            rows_X_train.append(df_dev.iloc[train_idx][feature_cols].to_numpy(dtype=np.float32))
            rows_X.append(df_dev.iloc[split_idx][feature_cols].to_numpy(dtype=np.float32))
            rows_y.append(df_dev.iloc[split_idx]["__label"].map(_CATEGORY_TO_IDX).to_numpy(dtype=np.int64))
            rows_dev.append(np.full(len(split_idx), device))

        if not rows_X:
            raise FileNotFoundError(
                f"No N-BaIoT device data found under {base}. "
                "Expected data/n_baiot/<device>/benign_traffic.csv and gafgyt_attacks/*.csv"
            )

        X = np.concatenate(rows_X, axis=0)
        y = np.concatenate(rows_y, axis=0)
        device_ids = np.concatenate(rows_dev, axis=0)
        X_train = np.concatenate(rows_X_train, axis=0)

        # Standardize using train-split stats only (fit on X_train regardless of
        # self.train, then applied to whichever split this instance returns) so
        # train and test features share the same scale -- previously each split
        # was standardized with its own stats, silently corrupting evaluation.
        # float64 accumulation avoids overflow on the large-scale N-BaIoT features
        # (e.g. "weight"/"magnitude") when summed over millions of rows.
        mean = X_train.astype(np.float64).mean(axis=0, keepdims=True)
        std = X_train.astype(np.float64).std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X = ((X.astype(np.float64) - mean) / std).astype(np.float32)

        self._dataset = NBaIoTDataset(X, y, device_ids)

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_adapter = NBaIoTAdapter(root=self.root, train=False, seed=self.seed, test_fraction=self.test_fraction)
        return DataLoader(test_adapter.dataset, batch_size=batch_size, shuffle=shuffle)

    def get_client_loaders(self, num_clients, strategy="iid", batch_size=64, seed=0, **strategy_args):
        if strategy.lower() != "device":
            return super().get_client_loaders(num_clients, strategy, batch_size, seed, **strategy_args)

        from torch.utils.data import Subset

        device_ids = self.dataset.device_ids
        parts = self._partition_by_device(device_ids, num_clients, seed=seed)
        return {
            cid: DataLoader(Subset(self.dataset, idxs), batch_size=batch_size, shuffle=True)
            for cid, idxs in parts.items()
        }

    @staticmethod
    def _partition_by_device(device_ids, num_clients, seed=0):
        rng = np.random.RandomState(seed)
        unique_devices = sorted(np.unique(device_ids).tolist())
        n_dev = len(unique_devices)
        client_indices = {i: [] for i in range(num_clients)}

        if num_clients <= n_dev:
            # Multiple devices grouped round-robin onto each client.
            for d_idx, dev in enumerate(unique_devices):
                cid = d_idx % num_clients
                client_indices[cid].extend(np.where(device_ids == dev)[0].tolist())
        else:
            # Each device's samples are split into shards spread round-robin over clients.
            base_slots, extra = divmod(num_clients, n_dev)
            cid_cursor = 0
            for d_idx, dev in enumerate(unique_devices):
                n_slots = base_slots + (1 if d_idx < extra else 0)
                idxs = np.where(device_ids == dev)[0]
                rng.shuffle(idxs)
                for shard in np.array_split(idxs, max(n_slots, 1)):
                    client_indices[cid_cursor % num_clients].extend(shard.tolist())
                    cid_cursor += 1

        for cid in client_indices:
            rng.shuffle(client_indices[cid])
        return {cid: list(map(int, idxs)) for cid, idxs in client_indices.items()}
