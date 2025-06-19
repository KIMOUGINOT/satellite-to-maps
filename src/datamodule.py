import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.satellite_dataset import SatelliteDataset

class SatelliteDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_path = os.path.join(cfg.dataset, 'train')
        self.val_path = os.path.join(cfg.dataset, 'val')
        self.test_path = os.path.join(cfg.dataset, 'test')

    def setup(self, stage=None):
        self.train_dataset = SatelliteDataset(dataset_path=self.train_path, image_size=self.cfg.image_size)
        self.val_dataset = SatelliteDataset(dataset_path=self.val_path, image_size=self.cfg.image_size)
        self.test_dataset = SatelliteDataset(dataset_path=self.test_path, image_size=self.cfg.image_size)

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers
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
    
# if __name__ == "__main__":
#     from omegaconf import OmegaConf

#     cfg = OmegaConf.load("config/config.yaml")
#     datamodule = SatelliteDataModule(cfg.data)
