import torch
from torch import nn

class GiBUUTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Embedding
        self.register_buffer('pdg_list', torch.tensor(cfg['pdg_list']))
        self.input_embedding = nn.Embedding(**cfg['transformer']['embedding'])
        self.output_embedding = nn.Embedding(**cfg['transformer']['embedding'])
        
        # Econder
        layer_cfg = cfg['transformer']['layer']
        encoder_layer = nn.TransformerEncoderLayer(**layer_cfg)
        
        d_model = layer_cfg['d_model']
        encoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, norm=encoder_norm,
            **cfg['transformer']['encoder']
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(**layer_cfg)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, norm=decoder_norm,
            **cfg['transformer']['decoder']
        )
            
    def tokenize(self, pdgids):
        tokens = torch.searchsorted(self.pdg_list, pdgids)
        return tokens

    def embed(self, pdgids, feats, embedding):
        tokens = self.tokenize(pdgids)
        x_embd = torch.cat((embedding(tokens), feats), dim=-1)
        return x_embd
        
    def embed_input(self, pdgids, feats):
        return self.embed(pdgids, feats, self.input_embedding)

    def embed_output(self, pdgids, feats):
        return self.embed(pdgids, feats, self.output_embedding)
        
    def forward(self ):
        # TODO(2023-06-08 kvt) Run encoder + decoder inside forward()
        return NotImplemented
