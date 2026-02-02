from torchvision import transforms
from typing import Tuple




def get_transforms(dataset_name: str, image_size: Tuple[int,int]=None, train: bool=True):
    """Return torchvision transforms for common datasets.


    - CIFAR-style: 32x32
    - MNIST: 28x28
    """
    if dataset_name.lower() in ('cifar10','cifar100'):
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])


    if dataset_name.lower() == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if dataset_name.lower() == 'gtsrb':
        if train:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(10),       # Rotate by up to 10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
            ])

        else:
            return  transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
            ])
        
    if dataset_name.lower() == 'femnist':
        if train:
            return transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # Normalize for grayscale images
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])


    # fallback: basic transform (resize if requested)
    t = [transforms.ToTensor()]
    if image_size is not None:
        t.insert(0, transforms.Resize(image_size))
    return transforms.Compose(t)