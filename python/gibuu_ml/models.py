import yaml
import torch
import numpy as np

from gibuu_ml.cfg import prepare_cfg
from gibuu_ml.io import dataloader_factory
from gibuu_ml.net import GiBUUTransformer

import lightning.pytorch as pl

# A model for predicting particle type
class ParticleTypeModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.net = GiBUUTransformer(cfg)

        opt_cfg = cfg.setdefault('optimizer', {})
        opt_cfg.setdefault('lr', 1e-5)

        self.save_hyperparameters()
        
    def forward(self, batch):
        # Generate src (x_in), tgt (x_out), and masks
        x_in = self.net.embed_input(batch['src_eid'], batch['src_feat'])
        src_padding_mask = batch['src_padding_mask']
        
        # Shift x_out (target entering network), batch['tgt_eid']
        # (expected result), and relevant masks
        x_out = self.net.embed_output(
            batch['tgt_eid'][:,:-1], batch['tgt_feat'][:,:-1]
        )
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x_out.size(dim=1), device=x_out.device,
        )

        # shift padding the same way as x_out
        tgt_padding_mask = batch['tgt_padding_mask'][:,:-1]
        
        # Run network
        output = self.net(
            x_in, x_out,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return output

    @staticmethod
    def func_loss_part_id(batch, output, **kwargs):
        # Switch output shape from [batch_size, n_particles, n_classes] 
        # to [batch_size, n_classes, n_particles]

        loss = torch.nn.functional.cross_entropy(
            output['class_out'].swapaxes(1,2),
            batch['tgt_eid'][:,1:],
            ignore_index=0, 
            **kwargs
        )

        return loss

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.func_loss_part_id(batch, output)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self.hparams.cfg['optimizer']
        )
        return optimizer
