from .adapter import DatasetAdapter
from torchvision.datasets import FashionMNIST
from .transforms import get_transforms
from torch.utils.data import DataLoader


class FashionMNISTAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        # default transform if not provided
        if transform is None:
            transform = get_transforms("fashion_mnist", train=train)

        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)

    def load_dataset(self) -> None:
        self._dataset = FashionMNIST(
            root=self.root,
            train=self.train,
            download=True,
            transform=self.transform
        )

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_set = FashionMNIST(
            root=self.root,
            train=False,
            download=True,
            transform=get_transforms(dataset_name="fashion_mnist", train=False)
        )

        return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
