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

class GiBUUTransformer(nn.Module):
    '''
    Transformer model for GiBUU FSI.
    '''
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
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, norm=encoder_norm,
            **cfg['transformer']['encoder']
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(**layer_cfg)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, norm=decoder_norm,
            **cfg['transformer']['decoder']
        )

        # query token embedding
        self.query_embd = nn.Embedding(1, d_model)

        # output size prediction
        out_cfg = cfg['transformer']['output']
        self.max_out_size = out_cfg['max_size']
        self.predict_size = out_cfg.get('predict_size', False)
        if self.predict_size:
            self.size_predictor = nn.Linear(d_model, self.max_out_size)

    def encode(self, src_eid, src_feat, src_padding_mask=None):
        '''
        Embed input particles and pass to transformer encoder.

        Note: To utilize the output size predictor, a speical token is expected
        in the first input element.
        
        `bs` - batch size
        `ni` - input size
        `nf` - number of input features
        `nd` - embedding dim.

        Parameters: 
        -----------
        src_eid: tensor 
            Encoded particle IDs of shape `(bs, ni)`.

        src_feat: tensor
            Input features of shape `(bs, ni, nf)`.

        src_padding_mask: tensor, optional
            Boolean padding mask of shape `(bs, ni)`. Default: `None`.

        Returns:
        --------
        src_enc: tensor
            Encoded particles in embedding space of shape `(bs, ni, nd)`.
        memory: tensor
            Output of the transformer encoder of shape `(bs, ni, nd)`.
        '''

        src_enc = self.particle_encoder(src_eid, src_feat)
        memory = self.encoder(src_enc, src_key_padding_mask=src_padding_mask)
        return src_enc, memory

    def decode(
        self, tgt_in, memory, memory_padding_mask=None, tgt_padding_mask=None
    ):
        '''
        Run transformer decoder for the query tokens and memeory from
        transformer encoder.

        `bs` - batch size
        `ni` - input size
        `nt` - target size
        `nd` - embedding dim.

        Parameters:
        -----------
        tgt_in: tensor
            Query input of shape `(bs, nt, nd)`.
        memory: tensor
            Memory output from the transformer encoder of shape `(bs, ni, nd)`
        memory_padding_mask: tensor, optional
            Padding mask for memory of shape `(bs, ni)`. Default: `None`.
            Usually it is the same as `src_padding_mask`, see `encode()`.
        tgt_padding_mask, tensor, optional
            Padding mask for the target query of shape `(bs, nt)`.
            Default: `None`.

        Returns:
        --------
        out_xfrm: tensor
            Transformer decoder output of shape `(bs, nt, nd)`.
        '''

        out_xfmr = self.decoder(
            tgt_in, memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        return out_xfmr

    def forward(self, src_enc, memory=None, src_padding_mask=None, tgt_padding_mask=None):
        '''
        Pass input to through the transformer encoder (optional) and  decoder.

        `bs` - batch size
        `nt` - target size
        `nc` - number of classes for particle type
        `ns` - number of classes for output size

        Arguments:
        ----------
        src_enc: tensor
            Input in embedding shape, see `encode()`.
        memory: tensor, optional
            Memory from tranformer encoder output, see `encode()`.
            If not specified, generates from `encode(src_enc,...)`.
            Default: `None`.
        src_padding_mask: tensor, optional
            Input padding mask, see `encode()`. Default: `None`.
        tgt_padding_mask: tensor, optional
            Target padding mask, see `decode()`. Default: `None`.
            If `predict_size==True`, generates by `get_tgt_padding_mask` using
            the max predicted size.

        Returns:
        --------
        A dictionary of

        out_xfmr: tensor
            Transformer decoder output, see `decode()`.
        out_logit: tensor
            Logits for output class prediction of shape `(bs, nt, nc)`.
        out_feat: tensor
            Output particle features of shape `(bs, nt, nf)`.
            These are usually physics output after `ParticleDecoder`.
        memory: tensor, optional
            Transformer encoder output when not specified by input arguments.
        size_logit: tensor, optional
            Logits for output size prediction of shape `(bs, ns)`
        out_padding_mask: tensor, optional
            Output padding mask of shape `(bs, nt)` when `predict_size==True`.
        '''

        output = {}

        if memory is None:
            memory = self.encoder(src_enc, src_key_padding_mask=src_padding_mask)
            output['memory'] = memory

        if self.predict_size:
            size_logit = self.size_predictor(memory[:,0])
            output['size_logit'] = size_logit

            if tgt_padding_mask is None:
                n_outs = size_logit.argmax(dim=-1)
                tgt_padding_mask = self.get_tgt_padding_mask(
                    n_outs, device=src_enc.device
                )
                output['out_padding_mask'] = tgt_padding_mask

        if tgt_padding_mask is not None:
            batch_size, token_size = tgt_padding_mask.shape
        else:
            batch_size, token_size = len(memory), self.max_out_size

        tgt_in = self.get_query_input(batch_size, token_size)
        out_xfmr = self.decode(
            tgt_in, memory, src_padding_mask, tgt_padding_mask
        )
        out_logit, out_feat = self.particle_decoder(out_xfmr)

        output.update({
            'out_xfmr': out_xfmr,
            'out_logit': out_logit,
            'out_feat': out_feat
        })
        return output

    @staticmethod
    def get_tgt_padding_mask(tgt_sizes, device=None):
        '''
        Generate tpadding mask from a list of target sizes.

        `bs` - batch size
        `nt` - target size

        Arguments:
        ----------
        tgt_sizes: iterable[int]
            Sizes of target of shape `(bs,)`
        device: `torch device` or `int`, optional
            Device index. Default: `None`.

        Returns:
        --------
        mask: tensor
            Padding mask of shape `(bs, nt)`, where `nt = max(tgt_sizes)`.
        '''
    
        batch_size = len(tgt_sizes)
        token_size = max(tgt_sizes)
        mask = torch.full((batch_size, token_size), False, device=device)
        for b, n in enumerate(tgt_sizes):
            mask[b,n:] = True
        return mask

    def get_query_input(self, batch_size, token_size):
        '''
        Generate input query for transformer decoder.

        `bs` - batch size
        `nt` - target size
        `nd` - embedding dim.

        Arguments:
        ----------
        batch_size: int
            Batch size.
        token_size: int
            Number of target to query (i.e. `nt`).

        Returns:
        --------
        q_in: tensor
            Query in embeded space of shape `(bs, nt, nd)`.
        '''
        q_in = self.query_embd.weight.unsqueeze(0).repeat(
            batch_size, token_size, 1
        )
        return q_in

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
