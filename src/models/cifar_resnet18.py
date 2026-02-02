import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

   

def replace_batchnorm_with_groupnorm(module: nn.Module, num_groups: int = 32) -> None:
    """
    Recursively finds all BatchNorm2d layers in a module and replaces them
    with GroupNorm layers.

    Args:
        module (nn.Module): The module to modify.
        num_groups (int): The number of groups to use for GroupNorm. A common default is 32.
    """
    for name, child in module.named_children():
        # If the child is a BatchNorm2d layer, replace it
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            
            # Ensure num_groups is a divisor of num_channels
            if num_channels % num_groups != 0:
                # Find the largest divisor of num_channels that is <= num_groups
                # This is a robust fallback for channels that are not divisible by the default num_groups.
                g = num_groups
                while num_channels % g != 0 and g > 1:
                    g -= 1
                effective_num_groups = g
            else:
                effective_num_groups = num_groups

            # Create and set the new GroupNorm layer
            new_layer = nn.GroupNorm(num_groups=effective_num_groups, num_channels=num_channels)
            setattr(module, name, new_layer)
        
        # If the child has its own children, recurse
        elif len(list(child.children())) > 0:
            replace_batchnorm_with_groupnorm(child, num_groups)


class CifarNetGN(nn.Module):
    """
    A ResNet-18 model adapted for CIFAR-10 where all BatchNorm layers
    have been replaced with GroupNorm layers to make it suitable for
    Federated Learning.
    """
    def __init__(self, num_classes: int = 10):
        super(CifarNetGN, self).__init__()
        
        # 1. Load a pre-built ResNet-18 model (training from scratch)
        self.resnet = models.resnet18(weights=None)

        # 2. **Crucially, replace all BatchNorm layers with GroupNorm**
        replace_batchnorm_with_groupnorm(self.resnet)

        # 3. Modify the first convolutional layer for CIFAR-10's 32x32 images
        # (Standard ResNet is optimized for larger ImageNet images)
        # This makes it less aggressive in downsampling.
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # Remove the initial max pooling

        # 4. Replace the final fully connected layer for the correct number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
