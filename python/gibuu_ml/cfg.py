import numpy as np

SOS_TOKEN, EOS_TOKEN = 0, 0x7ff

def prepare_cfg(cfg):
    # embedding
    embedding_cfg = cfg['transformer']['embedding']
    
    # encoder/decoder layer
    d_model = embedding_cfg['embedding_dim'] + len(cfg['dataset']['feat_keys'])
    layer_cfg = cfg['transformer']['layer']
    layer_cfg['d_model'] = d_model    
