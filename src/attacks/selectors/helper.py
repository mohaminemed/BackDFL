"""helper functions for attacks module
Contains:
- ConvAutoEncoder: a convolutional autoencoder for image data (MNIST, FEMNIST, CIFAR10, GTSRB)
- ResNet18Encoder (pretrained backbone)
- train_autoencoder() and extract_latents() utilities
- extract_resnet_features() utility
- save/load helpers
"""


from typing import Tuple, Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


# --- ConvAutoencoder -------------------------------------------------------
class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder suitable for small images.

    Parameters
    ----------
    in_channels: int
        1 for grayscale (MNIST/FEMNIST) or 3 for RGB (CIFAR/GTSRB)
    latent_dim: int
        Dimensionality of the latent embedding
    base_channels: int
        Number of channels for first conv block (doubles in second block)
    input_size: int
        Height/width of the square input (28 for MNIST, 32 for CIFAR, etc.)
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 64, base_channels: int = 32, input_size: int = 28):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.input_size = input_size

        # encoder
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self._enc_flatten_dim = (base_channels * 2) * (input_size // 4) * (input_size // 4)
        self.enc_fc = nn.Linear(self._enc_flatten_dim, latent_dim)

        # decoder
        self.dec_fc = nn.Linear(latent_dim, self._enc_flatten_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(base_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out_act = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        z = self.enc_fc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        x = self.dec_fc(z)
        x = x.view(batch, self.base_channels * 2, self.input_size // 4, self.input_size // 4)
        x = F.relu(self.dec_deconv1(x))
        x = self.dec_deconv2(x)
        x = self.out_act(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# --- ResNet encoder --------------------------------------------------------
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    """ResNet18-based encoder. Returns feature vector for an input image.

    Parameters
    ----------
    pretrained: bool
        whether to use ImageNet pretrained weights
    out_dim: int
        dimension to project features to (if !=512, a linear layer is applied)
    adapt_first_conv: bool
        if True and input_channels != 3, reinitializes first conv to match input channels
    input_channels: int
        number of input channels (1 for grayscale)
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 512, adapt_first_conv: bool = False, input_channels: int = 3):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        if adapt_first_conv and input_channels != 3:
            conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                            stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
        # remove final fc
        self.backbone.fc = nn.Identity()
        self.out_dim = out_dim
        self.project = nn.Identity() if out_dim == 512 else nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.project(feats)


# --- training and extraction utilities ------------------------------------

def train_autoencoder(
    model: ConvAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_every: int = 200,
):
    """Train the ConvAutoencoder. Expects dataloader to yield (x, y) where x is in [0,1]."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        it = 0
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            opt.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            it += 1
            if (i + 1) % log_every == 0:
                print(f"AE epoch {ep+1}/{epochs} iter {i+1} avg_loss={total_loss/it:.6f}")
        if it > 0:
            print(f"AE epoch {ep+1}/{epochs} finished avg_loss={total_loss/it:.6f}")
    model.eval()


def extract_latents(
    model: ConvAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    return_labels: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run encoder on dataloader and return (latents, labels).

    latents: (N, latent_dim) numpy array
    labels: (N,) numpy array if return_labels True
    """
    model.to(device)
    model.eval()
    latents = []
    labels = [] if return_labels else None
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0].to(device), batch[1]
            else:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                y = None
            z = model.get_latent(x)
            latents.append(z.cpu().numpy())
            if return_labels:
                if y is None:
                    raise RuntimeError("Requested labels but dataloader yields no labels")
                labels.append(np.array(y))
    latents = np.concatenate(latents, axis=0)
    if return_labels:
        labels = np.concatenate(labels, axis=0)
        return latents, labels
    return latents, None


def extract_resnet_features(
    model: ResNet18Encoder,
    dataloader: DataLoader,
    device: torch.device,
    return_labels: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract features from ResNet encoder.

    Returns: (N, feat_dim) numpy array and optional labels.
    """
    model.to(device)
    model.eval()
    feats = []
    labels = [] if return_labels else None
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0].to(device), batch[1]
            else:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                y = None
            out = model(x)
            feats.append(out.cpu().numpy())
            if return_labels:
                if y is None:
                    raise RuntimeError("Requested labels but dataloader yields no labels")
                labels.append(np.array(y))
    feats = np.concatenate(feats, axis=0)
    if return_labels:
        labels = np.concatenate(labels, axis=0)
        return feats, labels
    return feats, None


# --- save / load helpers --------------------------------------------------

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_autoencoder(model: ConvAutoencoder, path: str, device: torch.device) -> ConvAutoencoder:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_resnet_encoder(model: ResNet18Encoder, path: Optional[str], device: torch.device) -> ResNet18Encoder:
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

