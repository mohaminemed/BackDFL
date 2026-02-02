from .adapter import DatasetAdapter
from torchvision.datasets import EMNIST
from .transforms import get_transforms
from typing import Optional
from torch.utils.data import DataLoader

class FEMNISTAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        # default transform if not provided
        if transform is None:
            transform = get_transforms("femnist", train=train)
        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)

    def load_dataset(self) -> None:
        self._dataset = EMNIST(
        root=self.root,
        split='byclass', # This split has 62 classes (digits, upper, lower)
        train=self.train,
        download=True,
        transform=self.transform
        )
        
    def get_test_loader(self, batch_size = 256, shuffle = False):
        test_set = EMNIST(root=self.root, split='byclass', download=True, train=False, transform=get_transforms(dataset_name="femnist", train=False))
        return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    
