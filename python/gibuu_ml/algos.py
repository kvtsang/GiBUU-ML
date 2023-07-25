import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

def max_bipartite_match(
    out_logit, out_feat, tgt_label, tgt_feat, tgt_padding_mask,
    return_numpy=False, device=None
):

    src_idx = []
    tgt_idx = []
    batch_idx = []
    tgt_sizes = torch.count_nonzero(~tgt_padding_mask, dim=-1).cpu()

    # loop batch
    for ib, size in enumerate(tgt_sizes):
        tgt_mask = ~tgt_padding_mask[ib]
        tgt_mask_idx = np.where(tgt_mask.cpu())[0]

        # ----------------------------
        # cost for type classification
        # approx. as -prob(tgt_label)
        # ----------------------------
        out_prob = out_logit[ib].softmax(-1)
        cost_cls = -out_prob[:,tgt_label[ib,tgt_mask]]

        # ----------------------------------------------------
        # cost for output features (e.g. position, 4-momentum)
        # ----------------------------------------------------
        cost_feat = torch.cdist(
            out_feat[ib], tgt_feat[ib, tgt_mask], p=1,
        )

        # -------------------------------------------
        # match pred. to target using Hungarian algo.
        # -------------------------------------------
        cost = cost_cls**2 + cost_feat**2
        #idx = linear_sum_assignment(cost.cpu())

        # FIXME(2023-07-24 kvt) ad-hoc fix for nan cost values
        idx = linear_sum_assignment(torch.nan_to_num(cost.cpu()))
        
        src_idx.append(idx[0])
        tgt_idx.append(tgt_mask_idx[idx[1]]) # map masked idx to tgt idx
        batch_idx.append(np.full(size,ib))
    
    src_idx = np.concatenate(src_idx)
    tgt_idx = np.concatenate(tgt_idx)
    batch_idx = np.concatenate(batch_idx)
    
    if return_numpy:
        return batch_idx, src_idx, tgt_idx
    
    as_tensor = lambda x : torch.as_tensor(x, dtype=torch.long, device=device)
    return as_tensor(batch_idx), as_tensor(src_idx), as_tensor(tgt_idx) 
