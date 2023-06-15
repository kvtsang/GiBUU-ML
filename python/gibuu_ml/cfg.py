import numpy as np

INVALID_GEN = -1
INVALID_TOKEN = 0
SOS_TOKEN, EOS_TOKEN = -999999, 999999

def prepare_cfg(cfg):
    
    # insert SOS and EOS tokens to pdg_list
    pdg_list = np.concatenate([
       [SOS_TOKEN], sorted(cfg['pdg_list']), [EOS_TOKEN]
    ])
    cfg['pdg_list'] = pdg_list
    
    # embedding
    embedding_cfg = cfg['transformer']['embedding']
    min_embeddings = len(cfg['pdg_list'])
    n_embeddings = embedding_cfg.setdefault(
        'num_embeddings', min_embeddings
    )
    assert n_embeddings>=min_embeddings, \
        f'num_embeddings should be >= {min_embeddings}'
    
    # encoder/decoder layer
    d_model = embedding_cfg['embedding_dim'] + len(cfg['dataset']['feat_keys'])
    
    layer_cfg = cfg['transformer']['layer']
    layer_cfg['d_model'] = d_model    
