from .adapter import DatasetAdapter
from torchvision.datasets.cifar import CIFAR10
from .transforms import get_transforms
from typing import Optional
from torch.utils.data import DataLoader

class CIFAR10Adapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        # default transform if not provided
        if transform is None:
            transform = get_transforms("cifar10", train=train)
        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)   
    
    def load_dataset(self) -> None:
        self._dataset = CIFAR10(root=self.root, train=self.train, transform=self.transform, download=self.download)

    def get_test_loader(self, batch_size = 256, shuffle = False):
        test_set = CIFAR10(root=self.root, train=False , download=True, transform=get_transforms(dataset_name="cifar10", train=False))
        return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)