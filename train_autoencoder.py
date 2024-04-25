import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import callbacks as plc

from model import EncoderDecoder, AutoEncoder
from data import DataInterface

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torch import nn
from utils import get_encoder, get_autodecoder, get_dataset


def load_callbacks(cfg: DictConfig):
    callbacks = []

    callbacks.append(
        plc.ModelCheckpoint(
            monitor="val_acc",
            filename="best-{epoch:02d}-{val_acc:.3f}",
            dirpath=f"./resources/models/{cfg.dataset.name}_{cfg.encoder.name}_{cfg.decoder.name}",
            save_weights_only=True,
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    )

    callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def get_loss_function(loss):
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss == "mse":
        return nn.MSELoss()
    elif loss == "l1":
        return nn.L1Loss()

    raise ValueError("Unknown loss function")


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    train_data, val_data = get_dataset(cfg)
    datamodule = DataInterface(
        train_set=train_data, test_set=val_data, **cfg.datamodule
    )
    encoder = get_encoder(cfg)
    decoder = get_autodecoder(cfg)
    reconstruciton_loss = get_loss_function(cfg.reconstruction_loss)
    sparse_penalty = get_loss_function(cfg.sparse_penalty)
    model = AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruciton_loss,
        sparse_penalty=sparse_penalty,
        lamda=cfg.lamda,
        cfg=cfg,
    )
    config = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(project="thesis", name=cfg.name, config=config)
    trainer = Trainer(
        logger=logger,
        callbacks=load_callbacks(cfg),
        max_epochs=cfg.epoch,
        devices=cfg.devices,
        precision="16-mixed",
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
