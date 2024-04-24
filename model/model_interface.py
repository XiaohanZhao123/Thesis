import torch
from torch import nn 
from spikingjelly.activation_based import layer, neuron, surrogate, functional
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch import optim
from torch.optim import lr_scheduler



class EncoderDecoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss, cfg: DictConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.cfg = cfg
        
    def forward(self, x):
        functional.reset_net(self.encoder)
        functional.reset_net(self.decoder)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.mean(dim=0)
    
    def training_step(self, batch, batch_idx):
        img, labels = batch
        output = self(img)
        loss = self.loss(output, labels)
        acc = (output.argmax(dim=-1) == labels).float().mean()
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        img, labels = batch
        output = self(img)
        loss = self.loss(output, labels)
        acc = (output.argmax(dim=-1) == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return (loss, acc)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name
        optimizer_cls = getattr(optim, optimizer_name)
        assert optimizer_cls is not None, f'Optimizer {optimizer_name} not found'
        
        optimizer = optimizer_cls(self.parameters(), **self.cfg.optimizer.kwargs)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        
        return [optimizer], [scheduler]
    