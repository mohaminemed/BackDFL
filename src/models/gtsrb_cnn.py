import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRB_CNN(nn.Module):
    def __init__(self, num_classes):
        super(GTSRB_CNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), # 32x32x3 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32x32x32 -> 16x16x32
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16x16x64 -> 8x8x64
            nn.Dropout(0.25)
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the feature maps
        x = self.fc_layers(x)
        return x