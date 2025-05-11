import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from satellite_dataset import SatelliteDataset

class SatelliteDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        train_path = os.path.join(cfg.dataset, 'train')
        val_path = os.path.join(cfg.dataset, 'val')
        test_path = os.path.join(cfg.dataset, 'test')

        self.train_dataset = SatelliteDataset(dataset_path=train_path, image_size=self.cfg.image_size)
        self.val_dataset = SatelliteDataset(dataset_path=val_path, image_size=self.cfg.image_size)
        self.test_dataset = SatelliteDataset(dataset_path=test_path, image_size=self.cfg.image_size)

    def train_dataloader(self):
        if len(self.test_dataset) == 0:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        if len(self.test_dataset) == 0:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
    
if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/config.yaml")
    datamodule = SatelliteDataModule(cfg.data)
