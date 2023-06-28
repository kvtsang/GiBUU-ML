import torch
from torch import nn

class GiBUUTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Embedding
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
        self.out = nn.Linear(d_model, cfg['transformer']['num_classes'])
            
    def embed(self, enc_ids, feats, embedding):
        x_embd = torch.cat((embedding(enc_ids), feats), dim=-1)
        return x_embd
        
    def embed_input(self, src_eid, src_feat):
        return self.embed(src_eid, src_feat, self.input_embedding)

    def embed_output(self, tgt_eid, tgt_feat):
        return self.embed(tgt_eid, tgt_feat, self.output_embedding)
        
        
    def forward(
        self, src, tgt, 
        src_mask=None, tgt_mask=None, memory_mask=None, 
        src_key_padding_mask=None, tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        """
        src: input that has already been embedded 
        src_mask: padding mask for src
        """

        results = dict()

        # encode      
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        results["memory"] = memory
        
        # decode
        x_out = self.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        results["decoder_out"] = x_out
        
        # convert to final outputs (n_classes)
        output = self.out(x_out)
        results["class_out"] = output
        
        return results
