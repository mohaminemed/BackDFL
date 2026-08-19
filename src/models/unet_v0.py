import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    """
    A helper function that creates a sequence of two convolutional layers,
    each followed by batch normalization and a ReLU activation.
    This is a standard building block for U-Net architectures.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    """
    A configurable U-Net implementation, suitable for the IBA trigger generator.
    The number of input and output channels can be specified.
    """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        # --- Encoder Path (Down-sampling) ---
        self.dconv_down1 = double_conv(in_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Decoder Path (Up-sampling) ---
        # The number of input channels for the decoder blocks is the sum of the
        # up-sampled feature map and the corresponding feature map from the encoder path.
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)

        # --- Final Output Layer ---
        # A 1x1 convolution to map the final feature map to the desired number of output channels.
        self.conv_last = nn.Conv2d(64, out_channel, 1)
        # Tanh activation to ensure the output (perturbation) is in the range [-1, 1].
        self.tanh = nn.Tanh()

    def forward(self, x):
        # --- Encoder ---
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)

        # --- Decoder with Skip Connections ---
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1) # Skip connection from conv3

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1) # Skip connection from conv2

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1) # Skip connection from conv1
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self.tanh(out)



class FEMNISTAutoencoder(nn.Module):
    """
    A lightweight convolutional autoencoder, suitable as a generator for
    the IBA attack on MNIST and FEMNIST datasets.

    This is a more efficient alternative to a full U-Net for simpler datasets.
    """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh() # Ensure output is in [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x