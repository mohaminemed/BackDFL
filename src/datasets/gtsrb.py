from .adapter import DatasetAdapter
from torchvision.datasets import GTSRB
from .transforms import get_transforms
from typing import Optional
from torch.utils.data import DataLoader

class GTSRBAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        # default transform if not provided
        if transform is None:
            transform = get_transforms("gtsrb", train=train)
        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)

    def load_dataset(self) -> None:
        self._dataset = GTSRB(root=self.root, split="train", transform=self.transform, download=self.download)

    def get_test_loader(self, batch_size = 256, shuffle = False):
        test_set = GTSRB(root=self.root, split='test', download=True, transform=get_transforms(dataset_name="gtsrb", train=False))
        return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
