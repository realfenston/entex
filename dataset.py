import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .common import ESM_TOKENIZER


class StrDataset(Dataset):
    def __init__(self, data_config):
        super(StrDataset, self).__init__()
        self.dataset_path = data_config.path
        self.samples = []

        with open(self.dataset_path, 'r') as data_f:
            lines = data_f.readlines()

        for line in tqdm(lines):
            sample = json.loads(line)
            if len(sample['coords']) == len(sample['seq']):
                sample['coords'] = [residue[:3] for residue in sample['coords']] #excluding oxygen atom on main-chain
                self.samples.append(sample)
            
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def featurize(batch, max_seq_length=512):
        B = len(batch)
        L_max = max([len(b['seq']) for b in batch])
        X = np.zeros([B, L_max, 3, 3])
        S = np.zeros([B, L_max], dtype=np.int32)

        for i, sample in enumerate(batch):
            l = len(sample['seq'])
            x = sample['coords']
            x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) #[atom, 3, 3]
            X[i,:,:,:] = x_pad

            # Convert to labels
            indices = np.array(ESM_TOKENIZER.encode(sample['seq'], add_special_tokens=False))
            S[i, :l] = indices

        mask = np.isfinite(np.sum(X,(2,3))).astype(np.int32)
        numbers = np.sum(mask, axis=1).astype(np.int32)

        S_new = np.zeros_like(S)
        X_new = np.zeros_like(X)+np.nan
        for i, n in enumerate(numbers):
            X_new[i,:n,::] = X[i][mask[i]==1]
            S_new[i,:n] = S[i][mask[i]==1]
        
        X = X_new
        S = S_new

        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X,(2,3))).astype(np.int32)
        X[isnan] = 0.0

        #seq, structure crop
        L = S.shape[1]
        if L > max_seq_length:
            X = X[:, :max_seq_length, ...]
            S = S[:, :max_seq_length]
            mask = mask[:, :max_seq_length]

        return {
            "name": [b['name'] for b in batch],
            "X": torch.from_numpy(X).to(torch.float32),
            "S": torch.from_numpy(S).to(torch.long),
            "mask": torch.from_numpy(mask).to(torch.long),
        }
    

if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    #import ipdb; ipdb.set_trace()
    with open('./entex/configs/entex_test.yml', 'r') as f:
        config =  EasyDict(yaml.safe_load(f))
    
    dataset_config = config.dataset
    dataset = StrDataset(dataset_config)
    print(len(dataset))

    StrDataset.featurize([dataset[0], dataset[1]])