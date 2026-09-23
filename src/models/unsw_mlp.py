import torch.nn as nn
import torch.nn.functional as F


class UNSW_MLP(nn.Module):
    """MLP for UNSW-NB15 tabular features (10-class attack category)."""

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
