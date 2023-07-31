import torch
from torch import nn

def gen_mlp(
    in_features, hidden_features, hidden_layers, out_features,
    Activation=nn.LeakyReLU, bias=True
):
    net = [
        nn.Linear(in_features, hidden_features, bias=bias),
        Activation(),
    ]
    
    for i in range(hidden_layers):
        net.extend([
            nn.Linear(hidden_features, hidden_features, bias=bias),
            Activation(),
        ])
        
    net.append(nn.Linear(hidden_features, out_features, bias=bias))
    return nn.Sequential(*net)

class ParticleEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.embedding = nn.Embedding(**cfg['embedding'])
        self.encoder = gen_mlp(**cfg['particle_encoder'])

    def forward(self, encoded_ids, feats):
        x_out = self.embedding(encoded_ids) + self.encoder(feats)
        return x_out

class ParticleDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        dec_cfg = cfg['particle_decoder']
        in_features = dec_cfg['in_features']
        out_features = dec_cfg['out_features']
        num_classes = dec_cfg['num_classes']
        max_size = dec_cfg['max_size']
        
        self.part_type_cls = nn.Linear(in_features, num_classes)
        self.part_feat_dec = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        part_type = self.part_type_cls(x)
        part_feat = self.part_feat_dec(x)
        return part_type, part_feat

class GiBUUTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Particle Encoder + Decoder
        self.particle_encoder = ParticleEncoder(cfg)
        self.particle_decoder = ParticleDecoder(cfg)

        # Transformer Encoder
        layer_cfg = cfg['transformer']['layer']
        encoder_layer = nn.TransformerEncoderLayer(**layer_cfg)
        
        d_model = layer_cfg['d_model']
        encoder_norm = nn.LayerNorm(d_model)
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, norm=encoder_norm,
            **cfg['transformer']['encoder']
        )


    def forward(self, src_enc, src_padding_mask=None):
        out_enc = self.transformer(
            src_enc, src_key_padding_mask=src_padding_mask
        )
        out_logit, out_feat = self.particle_decoder(out_enc)

        res = {
            'out_enc' : out_enc,
            'out_logit' : out_logit,
            'out_feat' : out_feat,
        }
        return res

class SetCriterion(nn.Module):
    def __init__(self, num_classes, class_weight={}):

        super().__init__()

        weight = torch.ones(num_classes)
        for i, w in class_weight.items():
            weight[i] = w
        self.register_buffer('class_weight', weight)
    
    def loss_cls(self, out_logit, tgt_label, indices):

        batch_idx, src_idx, tgt_idx = indices
        
        # matched target label, shape as out_logit
        tgt_label_m = torch.zeros(
            out_logit.shape[:2], dtype=torch.long, device=tgt_label.device,
        )
        tgt_label_m[batch_idx, src_idx] = tgt_label[batch_idx, tgt_idx]
        
        loss = nn.functional.cross_entropy(
            out_logit.swapaxes(1,2), tgt_label_m, self.class_weight,
        )
        return loss
    
    def loss_feat(self, out_feat, tgt_feat, indices):

        batch_idx, src_idx, tgt_idx = indices
        
        loss = nn.functional.l1_loss(
            out_feat[batch_idx, src_idx], tgt_feat[batch_idx, tgt_idx]
        )
        return loss
        
    def forward(self, out_logit, out_feat, tgt_label, tgt_feat, indices):

        loss = {
            'loss_match_cls': self.loss_cls(out_logit, tgt_label, indices),
            'loss_match_feat': self.loss_feat(out_feat, tgt_feat, indices),
        }
        
        return loss
