"""
Useful Constants and Typings.
"""

import torch
from typing import Dict, Any

NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "EMNIST_BYCLASS": 62,
    "EMNIST_BALANCED": 47,
    "EMNIST_DIGITS": 10,
    "FEMNIST": 62,
    "TINYIMAGENET": 200,
    "SENTIMENT140": 2,
    "REDDIT": 50000,
    "GTSRB": 43,
    "FASHIONMNIST": 10,
    "HAR": 6,
    "NSLKDD": 5,
    "UNSW_NB15": 10
}

# Tabular (non-image) datasets: feature vectors instead of images. Only
# feature-space backdoor attacks (badnets, neurotoxin, scaling, tdfed) are
# supported for these, since image-specific attacks (a3fl, iba, dba, patch)
# assume spatial/pixel structure that tabular data does not have.
TABULAR_DATASETS = {"HAR", "NSLKDD", "UNSW_NB15", "NBAIOT"}

IMG_SIZE = {
    "CIFAR10": (32, 32, 3),
    "CIFAR100": (32, 32, 3),
    "MNIST": (28, 28, 1),
    "FASHIONMNIST": (28, 28, 1),
    "EMNIST_BYCLASS": (28, 28, 1),
    "EMNIST_BALANCED": (28, 28, 1),
    "EMNIST_DIGITS": (28, 28, 1),
    "FEMNIST": (28, 28, 1),
    "TINYIMAGENET": (64, 64, 3),
    "GTSRB": (32, 32, 3)
}

SEQ_LENGTH = {
    "HAR": (128, 9)  # 128 time steps, 9 channels
}

Metrics = Dict[str, float]
StateDict = Dict[str, torch.Tensor]
client_id = int
num_examples = int