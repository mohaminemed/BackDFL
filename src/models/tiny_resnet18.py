import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_TinyImageNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.model = resnet18(weights=None)  # no pretrained weights for 64×64
        # Replace first conv: kernel=3, stride=1, no maxpool
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        # Replace classifier
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


import torchvision.models as models

def ResNet18_ImageNet(num_classes=1000):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model
