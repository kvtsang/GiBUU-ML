import torch
from torch import nn

class GiBUUTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Embedding
        self.register_buffer('pdg_list', torch.tensor(cfg['pdg_list']))
        self.input_embedding = nn.Embedding(**cfg['transformer']['embedding'])
        self.output_embedding = nn.Embedding(**cfg['transformer']['embedding'])
        
        # Encoder
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
        
        # Last layers to convert to predictions
        self.out = nn.Linear(d_model, len(cfg['pdg_list']))
            
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
        
    def create_tgt_mask(self, tgt_size):
        """
        tgt_size: # of columns in tgt, e.g: 66
        
        Creates lower triangular square matrix of tgt_size, where bottom is 0 and top is -inf, to mask future elements.
        """
        return nn.Transformer.generate_square_subsequent_mask(tgt_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        src: input that has already been embedded 
        src_mask: padding mask for src
        """
        # TODO(2023-06-08 kvt) Run encoder + decoder inside forward()
        results = dict()
        # encode      
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        results["memory"] = memory
        
        # decode
        x_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        results["decode_output"] = x_out
        
        # convert to final outputs, softmax to #_of_pdgids
        output = self.act_out(self.out(x_out))
        results["output"] = output
        
        return results