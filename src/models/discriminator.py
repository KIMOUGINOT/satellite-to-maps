import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# PatchGAN Discriminator                                                     
###############################################################################

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN as in pix2pix.
    Input: concat(source_image, target_image) → 6 channels
    Output: N×N map of patch‐realism scores
    """
    def __init__(self, in_channels=6, base_filters=64, n_layers=3):
        """
        Args:
            in_channels: number of input channels (for pix2pix, 3 for source + 3 for target)
            base_filters: number of filters in the first conv layer
            n_layers: how many down‐sampling blocks (beyond the first) we use
        """
        super().__init__()
        layers = []

        # 1) First layer: no normalization, halves spatial dims
        #    Conv → LeakyReLU
        layers.append(
            nn.Conv2d(in_channels, base_filters,
                      kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2) Down–sampling layers: progressively double filters
        nf = base_filters
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                nn.Conv2d(nf_prev, nf,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # 3) One more layer with stride=1: keep resolution, increase receptive field
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            nn.Conv2d(nf_prev, nf,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # 4) Final conv to 1-channel output (patch scores)
        layers.append(
            nn.Conv2d(nf, 1,
                      kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, src_img, tgt_img):
        # concatenate source and target along channels: [B,6,H,W]
        x = torch.cat([src_img, tgt_img], dim=1)
        # output is [B,1,H',W'] patch map
        return torch.sigmoid(self.model(x))
