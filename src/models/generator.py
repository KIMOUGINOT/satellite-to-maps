import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Helpers: down‐ and up‐sampling blocks with InstanceNorm & Leaky/ReLU        
###############################################################################

class DownBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU, with optional normalization."""
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [ nn.Conv2d(in_channels, out_channels,
                             kernel_size=4, stride=2, padding=1, bias=not normalize) ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    """TransposeConv → InstanceNorm → Dropout? → ReLU."""
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

###############################################################################
# U-Net Generator                                                              
###############################################################################

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        # Encoder: 8 levels (down1 .. down8)
        self.down1 = DownBlock(in_channels,    base_filters, normalize=False)  # 256 → 128
        self.down2 = DownBlock(base_filters,   base_filters*2)                  # 128 → 64
        self.down3 = DownBlock(base_filters*2, base_filters*4)                  # 64 → 32
        self.down4 = DownBlock(base_filters*4, base_filters*8)                  # 32 → 16
        self.down5 = DownBlock(base_filters*8, base_filters*8)                  # 16 → 8
        self.down6 = DownBlock(base_filters*8, base_filters*8)                  # 8 → 4
        self.down7 = DownBlock(base_filters*8, base_filters*8)                  # 4 → 2
        self.down8 = DownBlock(base_filters*8, base_filters*8, normalize=False) # 2 → 1 bottleneck

        # Decoder: 7 up blocks (up1 .. up7) plus final up
        self.up1 = UpBlock(base_filters*8,    base_filters*8, dropout=True)    # 1 → 2
        self.up2 = UpBlock(base_filters*16,   base_filters*8, dropout=True)    # 2 → 4
        self.up3 = UpBlock(base_filters*16,   base_filters*8, dropout=True)    # 4 → 8
        self.up4 = UpBlock(base_filters*16,   base_filters*8)                  # 8 → 16
        self.up5 = UpBlock(base_filters*16,   base_filters*4)                  # 16 → 32
        self.up6 = UpBlock(base_filters*8,    base_filters*2)                  # 32 → 64
        self.up7 = UpBlock(base_filters*4,    base_filters)                    # 64 → 128

        # Final layer: up to full resolution + Tanh
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8);       u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1);       u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2);       u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3);       u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4);       u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5);       u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6);       u7 = torch.cat([u7, d1], dim=1)

        return self.final(u7)

# Why U-Net in pix2pix?
# Detail Preservation: Skip links carry fine‐grained features directly from early layers to the decoder.

# Efficiency: A relatively small number of parameters (compared to, say, very deep residual nets) but still captures multi‐scale context.

# Stochasticity without z: Dropout during training is enough to produce varied outputs if desired—pix2pix omits an explicit noise vector 
# for simplicity.