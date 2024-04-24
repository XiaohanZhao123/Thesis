import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import callbacks as plc

from model import EncoderDecoder
from data import DataInterface

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torch import nn
from utils import get_encoder, get_decoder, get_dataset


def load_callbacks():
    callbacks = []
    callbacks.append(
        plc.EarlyStopping(monitor="val_acc", mode="max", patience=10, min_delta=0.001)
    )

    callbacks.append(
        plc.ModelCheckpoint(
            monitor="val_acc",
            filename="best-{epoch:02d}-{val_acc:.3f}",
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    )

    callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))

    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    train_data, val_data = get_dataset(cfg)
    datamodule = DataInterface(
        train_set=train_data,
        test_set=val_data,
        **cfg.datamodule
    )
    encoder = get_encoder(cfg)
    decoder = get_decoder(cfg)
    
    if cfg.loss == "cross_entropy":
        loss = nn.CrossEntropyLoss()
    elif cfg.loss == "mse":
        loss = nn.MSELoss()
    else:
        raise ValueError("Unknown loss function")

    model = EncoderDecoder(encoder, decoder, loss, cfg)
    config = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(project="thesis", name=cfg.name, config=config)
    trainer = Trainer(
        logger=logger,
        callbacks=load_callbacks(),
        max_epochs=cfg.epochs,
        devices=cfg.devices,
        precision="16-mixed",
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule.test_dataloader)
