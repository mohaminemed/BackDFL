from .adapter import DatasetAdapter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .transforms import get_transforms
from typing import Optional
import os
import urllib.request
import zipfile

class TinyImageNetAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform: Optional = None):
        if transform is None:
            transform = get_transforms("tinyimagenet", train=train)
        self.transform = transform
        self.root = root
        self.download = download
        self.train = train

        if download:
            self._download_tinyimagenet()

        super().__init__(root=root, train=train, download=download, transform=transform)

    def _download_tinyimagenet(self):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        data_folder = os.path.join(self.root, "tiny-imagenet-200")
        if os.path.exists(data_folder):
            print("TinyImageNet already downloaded.")
            return
        os.makedirs(self.root, exist_ok=True)
        print(f"Downloading TinyImageNet to {zip_path} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting TinyImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(zip_path)
        print("TinyImageNet ready.")

    def load_dataset(self):
        split = "train" if self.train else "val"
        self._dataset = ImageFolder(
            root=os.path.join(self.root, "tiny-imagenet-200", split),
            transform=self.transform
        )
        self._dataset.targets = [label for _, label in self._dataset.samples]

    def get_test_loader(self, batch_size=256, shuffle=False):
        test_set = ImageFolder(
            root=os.path.join(self.root, "tiny-imagenet-200", "val"),
            transform=get_transforms("tinyimagenet", train=False)
        )
        test_set.targets = [label for _, label in test_set.samples]
        return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

