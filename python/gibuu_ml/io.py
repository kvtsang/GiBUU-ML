import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader

class GiBUUStepDataset(Dataset):
    def __init__(self, filepath, feat_keys):
        super().__init__()
        self._fp = h5py.File(filepath, 'r')
        
        names = self._fp['src'].attrs['names']
        
        self._col_names = names
        self._col_mask = self._get_col_mask(names, feat_keys)
        self._features = names[self._col_mask]
    
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
        
    def __del__(self):
        self._fp.close()
        
    def __len__(self):
        return len(self._fp['src'])
    
    def __getitem__(self, idx):
        src = self._fp['src'][idx]
        src_mask = self._fp['src_mask'][idx]
        
        tgt = self._fp['tgt'][idx]
        tgt_mask = self._fp['tgt_mask'][idx]
        
        names = self._col_names
        col_mask = self._col_mask
        output = {
            'idx' : idx,
            'src_gen' : src[:, names=='gen'].astype(int).squeeze(),
            'src_pdgid' : src[:, names=='barcode'].astype(int).squeeze(),
            'src_feat' : src[:, col_mask],
            'src_mask' : src_mask,
            
            'tgt_gen' : tgt[:, names=='gen'].astype(int).squeeze(),
            'tgt_pdgid' : tgt[:, names=='barcode'].astype(int).squeeze(),
            'tgt_feat' : tgt[:, col_mask],
            'tgt_mask' : tgt_mask,
        }
        
        return output
    
def dataloader_factory(cfg):
    dataset = GiBUUStepDataset(**cfg['dataset'])
    dataloader = DataLoader(dataset, **cfg['dataloader'])
    return dataloader
