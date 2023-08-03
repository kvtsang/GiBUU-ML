import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from gibuu_ml.net import GiBUUTransformerEncoder, SetCriterion
from gibuu_ml.algos import max_bipartite_match
from gibuu_ml.particle import mask_real_bit

class GiBUUStepModelV2a(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        model_cfg = cfg['model']
        self.net = GiBUUTransformerEncoder(model_cfg)
        self.crit = SetCriterion(**model_cfg['set_criterion'])

        opt_cfg = cfg.setdefault('optimizer', {})
        self.lr = opt_cfg.setdefault('lr', 5e-2)

        self.save_hyperparameters()
        
    def forward(self, src, src_padding_mask=None):
        return self.net(src, src_padding_mask)

    def encode_and_forward(self, batch):
        src_enc = self.net.particle_encoder(
            batch['src_eid'], batch['src_feat']
        )

        output = self(src_enc, batch['src_padding_mask'])
        output['src_enc'] = src_enc
        return output

    def match(self, output, batch):
        indices = max_bipartite_match(
            output['out_logit'].detach(),
            output['out_feat'].detach(),
            batch['tgt_eid'],
            batch['tgt_feat'],
            batch['tgt_padding_mask'],
            device=self.device,
        )
        output['matching_indices'] = indices
        return indices

    def cal_loss_match(self, output, batch):
        if 'matching_indices' not in output:
            self.match(output, batch)

        loss = self.crit(
            output['out_logit'], 
            output['out_feat'], 
            batch['tgt_eid'],
            batch['tgt_feat'], 
            output['matching_indices'],
        )

        return loss
    
    def cal_loss_self(self, output, batch, prefix):
        x_enc = output.get(f'{prefix}_enc')
        if x_enc is None:
            x_enc = self.net.particle_encoder(
                batch[f'{prefix}_eid'], batch[f'{prefix}_feat'],
            )

        logit_dec, feat_dec = self.net.particle_decoder(x_enc)

        # ignore is_real bit (decoder doesn't predict real/pert)
        label = mask_real_bit(batch[f'{prefix}_eid'])
        loss_cls = F.cross_entropy(
            logit_dec.swapaxes(1,2), label, self.crit.class_weight
        )

        # ignore padding particles
        mask = ~batch[f'{prefix}_padding_mask']
        loss_feat = F.l1_loss(
            feat_dec[mask], batch[f'{prefix}_feat'][mask], reduction='none'
        )

        loss = {
            f'loss_{prefix}_cls' : loss_cls,
            f'loss_{prefix}_feat' : loss_feat.mean(),
        }
        return loss

    def forward_and_loss(self, batch, batch_idx=-1):
        output = self.encode_and_forward(batch)

        try:
            self.match(output, batch)
        except ValueError:
            torch.save(output, f'debug_output_{batch_idx}.pkl')
            torch.save(batch, f'debug_batch_{batch_idx}.pkl')

        loss = self.cal_loss_match(output, batch)
        loss.update(self.cal_loss_self(output, batch, 'src'))
        loss.update(self.cal_loss_self(output, batch, 'tgt'))

        loss['loss_match'] = loss['loss_match_cls'] + loss['loss_match_feat']
        loss['loss_self'] = 0.5 * (
            loss['loss_src_cls'] + loss['loss_src_feat']
            + loss['loss_tgt_cls'] + loss['loss_tgt_feat']
        )
        loss['loss'] = loss['loss_match']  + loss['loss_self']

        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.forward_and_loss(batch, batch_idx)
        for k,v in loss.items():
            self.log(k, v, prog_bar=k=='loss', on_epoch=True, sync_dist=True)

        return loss['loss']
    
    def validation_step(self, batch, batch_idx):
        output, loss = self.forward_and_loss(batch, batch_idx)
        for k,v in loss.items():
            self.log(f'val_{k}', v, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self.hparams.cfg['optimizer']
        )
        return optimizer

class GiBUUStepModelV2b(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        
        model_cfg = cfg['model']
        self.net = GiBUUTransformer(model_cfg)
        self.crit = SetCriterion(**model_cfg['set_criterion'])
        self.predict_size = cfg['transformer']['output'].get('predict_size', False)
        if self.predict_size:
            self.SOS = -9999

        opt_cfg = cfg.setdefault('optimizer', {})
        self.lr = opt_cfg.setdefault('lr', 5e-2)

        self.save_hyperparameters()
                                          
    def forward(self, src_enc, memory=None, src_padding_mask=None, tgt_padding_mask=None):
        return self.net(enc_src, memory, src_padding_mask, tgt_padding_mask)
        
    def encode_and_forward(self, batch):
        '''
        Embeds input particles and passes to transformer encoder. 
        If output size predictor is specified, prepends a SOS token to src_eid and modifies
        src_feat and src_padding_mask accordingly. 
        '''
        src_eid = batch['src_eid']
        src_feat = batch['src_feat']
        src_padding_mask = batch['src_padding_mask']
    
        if self.predict_size:
            batch_size = batch['src_eid'].size(dim=0)
            
            # prepend SOS token to src_eid for output size predictor 
            sos_eid = torch.full((batch_size,1), self.SOS)
            src_eid = torch.cat((sos_eid, src_eid), dim=1)
            
            # prepend SOS token to src_feat
            sos_feat = torch.full((batch_size,1,7), self.SOS)
            src_feat = torch.cat((sos_feat, src_feat), dim=1) 
            
            # prepend True to src_padding_mask to mask first slot used for output size predictor
            sos_pad = torch.full((batch_size,1), True)
            src_padding_mask = torch.cat((sos_pad, src_padding_mask), dim=1)
            
        src_enc, memory = self.net.encode(src_eid, src_feat, src_padding_mask) 
        
        # provide network true output size during training, so pass batch['tgt_padding_mask']
        output = self.net(src_enc, memory, src_padding_mask, batch['tgt_padding_mask'])
        output['src_enc'] = src_enc
        output['memory'] = memory
        return output
                                          
    def match(self, output, batch):
        indices = max_bipartite_match(
            output['out_logit'].detach(),
            output['out_feat'].detach(),
            batch['tgt_eid'],
            batch['tgt_feat'],
            batch['tgt_padding_mask'],
            device=self.device,
        )
        output['matching_indices'] = indices
        return indices
    
    def cal_loss_match(self, output, batch):
        if 'matching_indices' not in output:
            self.match(output, batch)

        loss = self.crit(
            output['out_logit'], 
            output['out_feat'], 
            batch['tgt_eid'],
            batch['tgt_feat'], 
            output['matching_indices'],
        )

        return loss
    
    def forward_and_loss(self, batch, batch_idx=-1):
        output = self.encode_and_forward(batch)

        try:
            self.match(output, batch)
        except ValueError:
            torch.save(output, f'debug_output_{batch_idx}.pkl')
            torch.save(batch, f'debug_batch_{batch_idx}.pkl')

        loss = self.cal_loss_match(output, batch)
        loss.update(self.cal_loss_self(output, batch, 'src'))
        loss.update(self.cal_loss_self(output, batch, 'tgt'))
        loss['loss_match'] = loss['loss_match_cls'] + loss['loss_match_feat']
        
        if self.predict_size:
            tgt_sizes = torch.count_nonzero(~batch['tgt_padding_mask'], dim=-1).cpu()
            loss['loss_size_logit'] = F.cross_entropy(output['size_logit'], tgt_sizes)
            
        loss['loss'] = loss['loss_match'] + loss['loss_size_logit']

        return output, loss
    
    def training_step(self, batch, batch_idx):
        output, loss = self.forward_and_loss(batch, batch_idx)
        for k,v in loss.items():
            self.log(k, v, prog_bar=k=='loss', on_epoch=True, sync_dist=True)

        return loss['loss']
    
    def validation_step(self, batch, batch_idx):
        output, loss = self.forward_and_loss(batch, batch_idx)
        for k,v in loss.items():
            self.log(f'val_{k}', v, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self.hparams.cfg['optimizer']
        )
        return optimizer