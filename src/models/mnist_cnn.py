import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):  # adjust classes for FEMNIST
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # input 1x28x28
        self.pool  = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # -> 6x24x24
        x = self.pool(x)             # -> 6x12x12
        x = F.relu(self.conv2(x))    # -> 16x8x8
        x = self.pool(x)             # -> 16x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MNISTNet(nn.Module):
    """Simple CNN for MNIST."""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Fashion_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Fashion_CNN, self).__init__()
        # Input images are 28x28x1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 28x28x32 -> 14x14x32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),# 14x14x32 -> 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # 14x14x64 -> 7x7x64
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the feature maps
        x = self.fc_layers(x)
        return x
    

class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNIST_CNN, self).__init__()
        # Input images are 28x28x1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x