import numpy as np

def prepare_cfg(cfg):
    
    # insert SOS and EOS tokens to pdg_list
    sos_token = -999999 
    eos_token = 999999
    pdg_list = np.concatenate([
        [sos_token], sorted(cfg['pdg_list']), [eos_token]
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
