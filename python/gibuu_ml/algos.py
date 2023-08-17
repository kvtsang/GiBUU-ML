import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

def max_bipartite_match(
    out_logit, out_feat, tgt_label, tgt_feat,
    out_padding_mask=None, tgt_padding_mask=None,
    return_numpy=False, ignore_null_pred=False, null_id=0, device=None
):
    '''
    Match output to target in permutations to minimize the cost function.

    Cost = -probability of class label + L1 of features of the matching pair

    `bs` = batch size
    `ni` = number of inputs/outputs
    `nt` = number of targets
    `nc` = number of classes
    `nf` = number of features

    Arguments:
    ----------
    out_logit: tensor
        output logits of shape `(bs, ni, nc)`.

    out_feat: tensor
        output features of shape `(bs, ni, nf)`.

    tgt_label: tensor
        target class label of shape `(bs, nt)`

    tgt_feat: tensor
        target features of shape `(bs, nt, nf)`.

    out_padding_mask: tensor, optional
        output padding mask of shape `(bs, ni)`. Default: `None`.

    tgt_padding_mask: tensor
        target padding mask of shape `(bs, nt)`. Default: `None`.

    return_numpy: bool
        return as numpy arrays. Default: `False`
        
    ignore_null_pred: bool
        ignore null predictions when matching. Default: `False`.
        
    null_id: int
        particle type ID corresponding to null particle. Default: `0`.

    device: torch.device
        output tensor device. Default: `None`

    Returns:
    --------
    batch_idx, src_idx, tgt_idx: tensors
        Indices for matched pairs of shape `(bs, min(ni,nt))`.
    '''
    
    gen_padding_mask = lambda x : torch.full(
        x.shape[:2], False, dtype=bool, device=x.device
    ) 

    if out_padding_mask is None:
        out_padding_mask = gen_padding_mask(out_logit)

    if tgt_padding_mask is None:
        tgt_padding_mask = gen_padding_mask(tgt_label)
        
    if ignore_null_pred:
        out_cls = out_logit.argmax(-1)
        out_null_mask = out_cls==null_id
        out_padding_mask |= out_null_mask

    src_idx = []
    tgt_idx = []
    batch_idx = []

    out_sizes = torch.count_nonzero(~out_padding_mask, dim=-1)
    tgt_sizes = torch.count_nonzero(~tgt_padding_mask, dim=-1)
    sizes = torch.stack((out_sizes, tgt_sizes)).min(dim=0).values.cpu()

    # loop batch
    for ib, size in enumerate(sizes):
        out_mask = ~out_padding_mask[ib]
        out_mask_idx = np.where(out_mask.cpu())[0]

        tgt_mask = ~tgt_padding_mask[ib]
        tgt_mask_idx = np.where(tgt_mask.cpu())[0]

        # ----------------------------
        # cost for type classification
        # approx. as -prob(tgt_label)
        # ----------------------------
        out_prob = out_logit[ib,out_mask].softmax(-1)
        cost_cls = -out_prob[:,tgt_label[ib,tgt_mask]]

        # ----------------------------------------------------
        # cost for output features (e.g. position, 4-momentum)
        # ----------------------------------------------------
        cost_feat = torch.cdist(
            out_feat[ib,out_mask], tgt_feat[ib,tgt_mask], p=1,
        )

        # -------------------------------------------
        # match pred. to target using Hungarian algo.
        # -------------------------------------------
        cost = cost_cls + cost_feat
        #idx = linear_sum_assignment(cost.cpu())

        # FIXME(2023-07-24 kvt) ad-hoc fix for nan cost values
        idx = linear_sum_assignment(torch.nan_to_num(cost.cpu()))

        # map masked idx to out/tgt idx
        src_idx.append(out_mask_idx[idx[0]])
        tgt_idx.append(tgt_mask_idx[idx[1]]) 
        batch_idx.append(np.full(size,ib))
    
    src_idx = np.concatenate(src_idx)
    tgt_idx = np.concatenate(tgt_idx)
    batch_idx = np.concatenate(batch_idx)
    
    if return_numpy:
        return batch_idx, src_idx, tgt_idx
    
    as_tensor = lambda x : torch.as_tensor(x, dtype=torch.long, device=device)
    return as_tensor(batch_idx), as_tensor(src_idx), as_tensor(tgt_idx) 
