import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from src.models import UNetGenerator, PatchDiscriminator

class Pix2PixLightning(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        beta1: float = 0.5,
        lambda_l1: float = 100.0,
        log_image_interval: int = 20,
    ):
        """
        LightningModule for pix2pix cGAN training.

        Args:
            generator: UNetGenerator instance.
            discriminator: PatchDiscriminator instance.
            lr: learning rate for both G and D.
            beta1: Adam beta1 hyperparameter.
            lambda_l1: weight for L1 reconstruction loss.
            log_image_interval: batches between image logging.
        """
        super().__init__()
        self.automatic_optimization = False
        self.generator = UNetGenerator()
        self.discriminator = PatchDiscriminator()
        self.lr = lr
        self.beta1 = beta1
        self.lambda_l1 = lambda_l1
        self.log_image_interval = log_image_interval

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        return [opt_d, opt_g]

    def training_step(self, batch, batch_idx):
        # Manual optimization with two optimizers
        opt_d, opt_g = self.optimizers()
        src, tgt = batch  # [B,3,256,256]

        # Generate fake image
        fake = self(src)

        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        # Real loss
        d_real = self.discriminator(src, tgt)
        loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        # Fake loss
        d_fake = self.discriminator(src, fake.detach())
        loss_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        # Total D loss
        d_loss = 0.5 * (loss_real + loss_fake)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log('d_loss', d_loss, prog_bar=True, on_step=True)

        # -----------------
        # Train Generator
        # -----------------
        opt_g.zero_grad()
        # Adversarial loss
        d_fake_for_g = self.discriminator(src, fake)
        loss_g_gan = F.binary_cross_entropy(d_fake_for_g, torch.ones_like(d_fake_for_g))
        # L1 reconstruction loss
        loss_g_l1 = F.l1_loss(fake, tgt) * self.lambda_l1
        g_loss = loss_g_gan + loss_g_l1
        self.manual_backward(g_loss)
        opt_g.step()
        self.log('g_loss', g_loss, prog_bar=True, on_step=True)

        # -----------------
        # Log images
        # -----------------
        if batch_idx % self.log_image_interval == 0:
            imgs = torch.cat([src, fake, tgt], dim=0)
            grid = make_grid(imgs, nrow=src.size(0), normalize=True)
            self.logger.experiment.add_image(
                'pix2pix_results', grid, self.global_step
            )

        return {'d_loss': d_loss, 'g_loss': g_loss}