from pytorch_lightning import LightningModule
import torch

class SatelliteModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()  #to log the config
        self.cfg = cfg

        self.model = ...
        self.criterion = ...
        self.metric = ...

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)

        loss = self.criterion(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)

        loss = self.criterion(preds, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)

        loss = self.criterion(preds, targets)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.train.lr_step, gamma=self.cfg.train.lr_gamma)
        return [optimizer], [scheduler]