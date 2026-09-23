import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .adapter import DatasetAdapter

# --- Standard NSL-KDD column layout (41 features + label + difficulty) ---
_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Fixed vocabularies (from the official NSL-KDD documentation) so that the
# one-hot encoding always has the same dimensionality regardless of which
# categories happen to appear in a given split.
_PROTOCOL_VALUES = ["tcp", "udp", "icmp"]
_SERVICE_VALUES = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard",
    "domain", "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger",
    "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784",
    "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin", "kshell",
    "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns",
    "netbios_ssn", "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump",
    "pop_2", "pop_3", "printer", "private", "red_i", "remote_job", "rje",
    "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat", "telnet",
    "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet",
    "whois", "X11", "Z39_50",
]
_FLAG_VALUES = ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"]

_CATEGORICAL_VOCAB = {
    "protocol_type": _PROTOCOL_VALUES,
    "service": _SERVICE_VALUES,
    "flag": _FLAG_VALUES,
}

_NUMERIC_COLS = [c for c in _COLUMNS if c not in _CATEGORICAL_COLS + ["label", "difficulty"]]

# Attack name -> attack category mapping (standard NSL-KDD taxonomy).
_ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "worm": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}
_CATEGORY_TO_IDX = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}


class NSLKDDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NSLKDDAdapter(DatasetAdapter):
    """
    NSL-KDD network intrusion detection dataset.

    Expected directory layout:
        data/nslkdd/KDDTrain+.txt
        data/nslkdd/KDDTest+.txt

    Feature layout after preprocessing (fixed across train/test):
        [ standardized numeric features (len(_NUMERIC_COLS)) |
          one-hot protocol_type (3) | one-hot service (70) | one-hot flag (11) ]

    Numeric features always occupy indices [0, num_numeric), which is what
    trigger implementations rely on to avoid corrupting the one-hot blocks.
    Target: 5-class attack category (normal, dos, probe, r2l, u2r).
    """

    num_numeric = len(_NUMERIC_COLS)

    def __init__(self, root="data", train=True, download=False, seed=42):
        self.seed = seed
        super().__init__(root=root, train=train, download=download, transform=None)

    def load_dataset(self) -> None:
        base = os.path.join(self.root, "nslkdd")
        train_path = os.path.join(base, "KDDTrain+.txt")
        test_path = os.path.join(base, "KDDTest+.txt")

        df_train = pd.read_csv(train_path, names=_COLUMNS, header=None)
        df_test = pd.read_csv(test_path, names=_COLUMNS, header=None)

        # Fit numeric scaling stats on train only, apply to both splits.
        mean = df_train[_NUMERIC_COLS].mean()
        std = df_train[_NUMERIC_COLS].std().replace(0, 1.0)

        df = df_train if self.train else df_test

        numeric = ((df[_NUMERIC_COLS] - mean) / std).to_numpy(dtype=np.float32)

        onehot_blocks = []
        for col in _CATEGORICAL_COLS:
            vocab = _CATEGORICAL_VOCAB[col]
            codes = pd.Categorical(df[col], categories=vocab)
            onehot = pd.get_dummies(codes).reindex(columns=vocab, fill_value=0).to_numpy(dtype=np.float32)
            onehot_blocks.append(onehot)

        X = np.concatenate([numeric] + onehot_blocks, axis=1)

        y = df["label"].str.lower().map(_ATTACK_CATEGORY_MAP).fillna("normal")
        y = y.map(_CATEGORY_TO_IDX).to_numpy(dtype=np.int64)

        self._dataset = NSLKDDDataset(X, y)

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_adapter = NSLKDDAdapter(root=self.root, train=False, seed=self.seed)
        return DataLoader(test_adapter.dataset, batch_size=batch_size, shuffle=shuffle)
