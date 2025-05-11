from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from src.module import Pix2PixLightning
from src.datamodule import SatelliteDataModule

cfg = OmegaConf.load("config/config.yaml")
logger = TensorBoardLogger(save_dir=cfg.train.output_dir, name="Pix2Pix")

model = Pix2PixLightning()
dataloader = SatelliteDataModule(cfg.data)

trainer = Trainer(
    max_epochs=cfg.train.max_epochs,
    accelerator="auto",
    devices="auto",
    logger=logger,
    default_root_dir=cfg.train.output_dir
)
trainer.fit(model, dataloader)