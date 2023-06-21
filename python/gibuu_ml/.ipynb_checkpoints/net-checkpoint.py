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
        self.out = nn.Linear(len(cfg['tgt_pdgid'][0]), len(cfg['pdg_list'])-1) # -1 to exclude start of sequesnce bc passed cfg is called after prepare_cfg
        self.act_out = nn.Softmax(dim=1)
            
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
        tgt_mask = torch.tril(torch.ones(tgt_size, tgt_size))
        tgt_mask = tgt_mask.float()
        tgt_mask = tgt_mask.masked_fill_(tgt_mask==0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill_(tgt_mask==1, float(0.0))
        
    def forward(self, pdgids, feats, src_padding_mask, tgt):
        # TODO(2023-06-08 kvt) Run encoder + decoder inside forward()
        # generate target mask
        tgt_mask = create_tgt_mask(tgt.size(dim=1))
            
        # encode
        x_in = net.embed_input(pdgids, feats)       
        memory = self.encoder(x_in, src_key_padding_mask=src_padding_mask)
        
        # decode
        x_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        
        # convert to final outputs, softmax to #_of_pdgids+1 (for end sequence) classes
        output = self.act_out(self.out(x_out))
        
        return output