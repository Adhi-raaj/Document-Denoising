"""
Denoising Autoencoder (U-Net) for Document Restoration
Lightweight architecture optimized for CPU inference and Colab training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling + skip connection + ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches (important for non-power-of-2 inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DenoisingUNet(nn.Module):
    """
    U-Net Denoising Autoencoder.

    Architecture:
        Encoder: 4 levels with features [32, 64, 128, 256]
        Bottleneck: 512 channels
        Decoder: 4 levels with skip connections
        Total params: ~7.7M (optimized for quality vs speed)

    For faster training/inference, use features=[16, 32, 64, 128] → ~1.9M params
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock(ch, feat))
            ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        decoder_features = list(reversed(features))
        bottleneck_ch = features[-1] * 2

        ch = bottleneck_ch
        for feat in decoder_features:
            self.decoder_blocks.append(UpBlock(ch, feat, feat))
            ch = feat

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encode
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        skips = list(reversed(skips))
        for i, dec in enumerate(self.decoder_blocks):
            x = dec(x, skips[i])

        return self.output_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(features: list = None, device: str = "cpu") -> DenoisingUNet:
    """Factory function to create and initialize model."""
    if features is None:
        features = [32, 64, 128, 256]
    model = DenoisingUNet(features=features)
    return model.to(device)


if __name__ == "__main__":
    model = get_model()
    print(f"Model parameters: {model.count_parameters():,}")
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
