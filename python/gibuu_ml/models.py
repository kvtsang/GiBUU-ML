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
        
    def training_step(self, batch, batch_idx):
        # Generate src (x_in), tgt (x_out), and masks
        x_in = self.net.embed_input(batch['src_eid'], batch['src_feat'])
        x_out = self.net.embed_output(batch['tgt_eid'], batch['tgt_feat'])
        src_padding_mask = batch['src_padding_mask']
        
        # Shift x_out (target entering network), batch['tgt_eid'] (expected result), and relevant masks
        x_out = x_out[:,:-1]
        expected_out = batch['tgt_eid'][:,1:]
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_out.size(dim=1), device=x_out.device)
        tgt_padding_mask = batch['tgt_padding_mask'][:,:-1] # shift padding the same way as x_out
        
        # Run network
        output = self.net(
            x_in, x_out, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask, 
            memory_key_padding_mask=src_padding_mask
        )
        
        # Switch output shape from [batch_size, n_particles, n_classes] to [batch_size, n_classes, n_particles]
        class_out = np.swapaxes(output['class_out'], 2, 1) 
            
        # Calculate loss
        loss = torch.nn.CrossEntropyLoss(ignore_index = 0)
        loss = loss(class_out, expected_out)
        
        self.log('loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self.hparams.cfg['optimizer']
        )
        return optimizer
