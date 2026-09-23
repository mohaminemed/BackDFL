import torch.nn as nn
import torch.nn.functional as F


class NBaIoT_MLP(nn.Module):
    """MLP for N-BaIoT tabular features (6-class: benign + 5 gafgyt subtypes)."""

    def __init__(self, input_dim=115, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
