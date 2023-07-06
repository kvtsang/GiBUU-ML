import h5py
import torch
import numpy as np
import numpy.lib.recfunctions as rfn

from .cfg import SOS_TOKEN, EOS_TOKEN
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

class GiBUUStepDataset(Dataset):
    def __init__(self, filepath, feat_keys, group_name, sort_tgt_by=None):
        super().__init__()
        self._fp = h5py.File(filepath, 'r')
        self._grp = self._fp[group_name]
        
        names = self._grp['src'].attrs['names']
        
        self._col_names = names
        self._col_mask = self._get_col_mask(names, feat_keys)
        self._col_idx = {key : i for i, key in enumerate(names)}
        self._features = names[self._col_mask]
        self._sort_tgt_by = sort_tgt_by

    @staticmethod
    def encode_id(gibuu_id, charge, is_real):

        #  --------------------------------
        # |   11    | 10 9 8 7 | 6 5 ... 0 | : bit
        #  --------------------------------
        # | is_real |   Q + 8  | GiBUU ID  | : content
        #  --------------------------------

        bits = gibuu_id \
            + ((charge+8) << 7) \
            + is_real * (1 << 11)
                
        return bits

    @staticmethod
    def decode_id(bits):
        gibuu_id = bits & 0x7f
        charge = ((bits >> 7) & 0xf) - 8
        is_real = (bits >> 11) & 0x1

        return gibuu_id, charge, is_real

    @staticmethod
    def _get_col_mask(names, cols):
        names = np.asarray(names, dtype='<U')
        cols = np.asarray(cols, dtype='U')
        
        idx_cmp = names[:,None] == cols[None,:]
        
        is_good = np.any(idx_cmp, axis=0)
        if not np.all(is_good):
            raise KeyError('Invalid column key(s)', cols[~is_good].tolist())
        
        idx = np.where(idx_cmp)[0]
        mask = np.full(len(names), False)
        mask[idx] = True
        return mask
        
    def _get_target(self, idx):
        tgt = self._grp['tgt'][idx]
        tgt_collision_mask = self._grp['tgt_collision_mask'][idx]
        tgt_padding_mask = self._grp['tgt_padding_mask'][idx]

        key = self._sort_tgt_by
        if key is None:
            return tgt, tgt_collision_mask, tgt_padding_mask

        i_col = self._col_idx[key]
        i_sorted = np.flip(np.argsort(tgt[:,i_col]))

        return tgt[i_sorted], tgt_collision_mask[i_sorted], tgt_padding_mask

    def __del__(self):
        self._fp.close()
        
    def __len__(self):
        return len(self._grp['info'])
    
    def __getitem__(self, idx):
        names = self._col_names
        col_mask = self._col_mask

        iID = self._col_idx['ID']
        iQ = self._col_idx['charge']

        # info
        info = self._grp['info']
        info_data = rfn.unstructured_to_structured(
            info[idx], names=info.attrs['names']
        )

        # real particles
        real= self._fp['real'][info_data['idx_real']]

        # concat real and pert. particles
        src = self._grp['src'][idx]

        src_feat = np.concatenate([real[:,col_mask], src[:,col_mask]])

        src_padding_mask = np.pad(
            self._grp['src_padding_mask'][idx], (len(real),0)
        )
        src_collision_mask = np.concatenate([
            self._grp['real_collision_mask'][idx], 
            self._grp['src_collision_mask'][idx],
        ])

        src_eid = np.concatenate([
            self.encode_id(
                real[:,iID].astype(int), real[:,iQ].astype(int), True
            ),
            self.encode_id(
                src[:,iID].astype(int), src[:,iQ].astype(int), False
            ),
        ])
        src_eid[src_padding_mask] = 0
        
        # get target (sort if needed)
        tgt, tgt_collision_mask, tgt_padding_mask = self._get_target(idx)

        # insert SOS and EOS to target
        tgt_padding_mask = np.pad(tgt_padding_mask, (1,1))
        tgt_collision_mask = np.pad(tgt_collision_mask, (1,1))
        tgt_feat = np.pad(tgt[:,col_mask], ((1,1),(0,0)))

        tgt_eid = np.concatenate([
            [SOS_TOKEN],
            self.encode_id(
                tgt[:,iID].astype(int), tgt[:,iQ].astype(int), False
            ),
            [EOS_TOKEN],
        ])
        tgt_eid[tgt_padding_mask] = 0

        # prepare output
        output = {
            'idx' : idx,

            'src_feat' : src_feat.astype(np.float32),
            'src_eid' : src_eid,
            'src_padding_mask' : src_padding_mask,
            'src_collision_mask' : src_collision_mask,
            
            'tgt_feat' : tgt_feat.astype(np.float32),
            'tgt_eid' : tgt_eid,
            'tgt_padding_mask' : tgt_padding_mask,
            'tgt_collision_mask' : tgt_collision_mask,
        }

        for key in info_data.dtype.names:
            output[key] = info_data[key].item()

        return output
    
def dataloader_factory(cfg):
    ds_cfg = cfg['dataset'].copy()

    fpatterns = ds_cfg.pop('filepath')

    if isinstance(fpatterns, str):
        fpatterns = [fpatterns]
    assert isinstance(fpatterns, list),  \
        "'filepath' must be a string / a list of strings'"

    files = []
    for expr in fpatterns:
        files.extend(glob(expr))

    datasets = [ GiBUUStepDataset(fpath, **ds_cfg) for fpath in files ]
    big_dataset = ConcatDataset(datasets)

    split_cfg = cfg.get('data_random_split')
    if split_cfg is None:
        dataloader = DataLoader(big_dataset, **cfg['dataloader'])
        val_dataloader = None
    else: 
        seed = split_cfg.get('seed')
        if seed is None:
            generator = torch.default_generator
        else:
            generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            big_dataset, split_cfg['lengths'], generator
        )

        dataloader = DataLoader(train_dataset, **cfg['dataloader'])
        val_dataloader = DataLoader(val_dataset, **cfg['val_dataloader'])

    return dataloader, val_dataloader
