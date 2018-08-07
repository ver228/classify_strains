#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:04:37 2018

@author: avelinojaver
"""
from flow_helper import add_folds, get_strains_ids, read_CeNDR_snps

import pandas as pd
import tables
import numpy as np
import os
import random
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

_divergent_set = ['N2',
 'CB4856',
 'DL238',
 'JU775',
 'MY16',
 'MY23',
 'CX11314',
 'ED3017',
 'EG4725',
 'LKC34',
 'JT11398',
 'JU258']


class SkelEmbeddingsFlow(Dataset):
    def __init__(self,
                 fname = '',
                 min_traj_size = 250,
                 sample_size = 22500,
                 fold_n_test = 0,
                 train_epoch_magnifier = 5,
                 is_divergent_set = False,
                 is_tiny = False,
                 is_balance_training = False,
                 unsampled_test = False,
                 ):
        
        assert os.path.exists(fname)
        
        self.fname = fname
        self.min_traj_size = min_traj_size
        self.sample_size = sample_size
        self.train_epoch_magnifier = train_epoch_magnifier
        self.is_balance_training = is_balance_training
        self.unsampled_test = unsampled_test
        
        with pd.HDFStore(fname) as fid:
            trajectories_ranges = fid['/trajectories_ranges']
            video_info = fid['/video_info']
            video_info['strain'] = video_info['strain'].str.strip(' ')
            video_info = add_folds(video_info)
            
            
            #add the size of each chuck in the video
            trajectories_ranges['size'] = trajectories_ranges['frame_fin'] - trajectories_ranges['frame_ini']
            
            #filter chucks that are too small
            trajectories_ranges = trajectories_ranges[trajectories_ranges['size'] >= min_traj_size]
            
            #get the total size of all the chucks in a video
            tot_ranges = trajectories_ranges.groupby('video_id').agg({'size':'sum'})
            video_info = video_info[tot_ranges['size'] >=sample_size]
            
            #only keep the strains that have at least 3 valid videos
            vid_per_strain = video_info['strain'].value_counts()
            ss = vid_per_strain.index[vid_per_strain.values > 2]
            video_info = video_info[video_info['strain'].isin(ss)]
            
        with tables.File(fname) as fid:
            skels_g = fid.get_node('/embeddings')
            self.embedding_size = skels_g.shape[1]
        
        if is_tiny:
            good = video_info['strain'].isin(['N2', 'CB4856'])
            video_info = video_info[good]
        if is_divergent_set:
            good = video_info['strain'].isin(_divergent_set)
            video_info = video_info[good]
        
        
        self.video_info = video_info
        self.video_traj_ranges = trajectories_ranges.groupby('video_id')
        self.fold_n_test = fold_n_test
        
        self.train_index = self.video_info.index[self.video_info['fold'] != self.fold_n_test].tolist()
        self.test_index = self.video_info.index[self.video_info['fold'] == self.fold_n_test].tolist()
        
        self.train()
    
    def train(self):
        self.is_train = True
        tot = len(self.train_index)*self.train_epoch_magnifier
        #
        
        #balance classes sampling
        if self.is_balance_training:
            strain_g = self.video_info.loc[self.train_index].groupby('strain').groups
            strains = list(strain_g.keys())
            self.valid_index = []
            while len(self.valid_index) < tot:
                random.shuffle(strains)
                for ss in strains:
                    ind = random.choice(strain_g[ss])
                    self.valid_index.append(ind)
        else:
            self.valid_index = [random.choice(self.train_index) for _ in range(tot)]
            
    def test(self):
        self.is_train = False
        self.valid_index = self.test_index
    
    
    def __getitem__(self, index):
        vid_id = self.valid_index[index]
        return self._get_data(vid_id)
        
    
    def _get_data(self, vid_id):
        if self.is_train or not self.unsampled_test:
            dat = self._sample_video(vid_id)
        else:
            dat = self._full_video(vid_id)
        return dat
    
    def __len__(self):
        return len(self.valid_index)
    
    def __iter__(self):
        for ind in range(len(self)):
            yield self[ind]
    
    def _full_video(self, vid_id):
        vid_ranges = self.video_traj_ranges.get_group(vid_id)
        #I might need to do a bit different if I am dealing with larger videos
        
        N = vid_ranges['size'].sum() + len(vid_ranges)
        with tables.File(self.fname) as fid:
            skels_g = fid.get_node('/embeddings')
            
            tot = 0
            sample_dat = np.zeros((N, self.embedding_size), dtype = np.float32)
            for _, row in vid_ranges.iterrows():
                dat = skels_g[row['row_ini']:row['row_fin']]
                sample_dat[tot:tot+dat.shape[0]] = dat
                tot += dat.shape[0] + 1  # the one is because I am leaving all zeros row between chunks
        return sample_dat

    
    def _sample_video(self, vid_id):
        #randomly select trajectories chuncks
        vid_ranges = self.video_traj_ranges.get_group(vid_id)
        tot = 0
        rows2read = []
        while tot < self.sample_size:
            row = vid_ranges.sample(1).iloc[0]
            row_ini = row['row_ini']
            row_fin = row['row_fin']
            size = row['size']
            
            remainder_t = self.sample_size - tot - 1
            top = min(remainder_t, size)
            bot = min(remainder_t, self.min_traj_size)
            size_r = random.randint(bot, top)
            
            ini_r = random.randint(row_ini, row_fin - size_r)
            
            
            rows2read.append((ini_r, ini_r + size_r))
            tot += size_r + 1 # the one is because I am leaving all zeros row between chunks
        
        #read embeddings using the preselected ranges
        fname = self.fname
        with tables.File(fname) as fid:
            skels_g = fid.get_node('/embeddings')
            
            tot = 0
            sample_dat = np.zeros((self.sample_size, self.embedding_size), dtype = np.float32)
            for ini, fin in rows2read:
                dat = skels_g[ini:fin]
                
                sample_dat[tot:tot+dat.shape[0]] = dat
                tot += dat.shape[0] + 1  # the one is because I am leaving all zeros row between chunks
        return sample_dat

#%%
class SkelTrainer(SkelEmbeddingsFlow):
    def __init__(self, return_label = True, return_snp = False, **argkws):
        super().__init__(**argkws)
        
        self._snps = read_CeNDR_snps()
        self._strain_dict = get_strains_ids(self._snps)
        
        valid_strains = self._snps.columns[4:].tolist()
        self._snps[valid_strains] = (self._snps[valid_strains]>0).astype(np.float32)
        {k:ii for ii, k in enumerate(valid_strains)}
        
        self.video_info = self.video_info[self.video_info['strain'].isin(valid_strains)]
        self.train_index = self.video_info.index[self.video_info['fold'] != self.fold_n_test]
        self.test_index = self.video_info.index[self.video_info['fold'] == self.fold_n_test]

        self.return_label = return_label
        self.return_snp = return_snp
        
        self.train()
    
    @property
    def num_classes(self):
        if self.return_label:
            return len(self._strain_dict)
        elif self.return_snp:
            return self._snps.shape[0]
    
    
    def __getitem__(self, index):
        vid_id = self.valid_index[index]
        
        print(vid_id)
        strain = self.video_info.loc[vid_id, 'strain']
        
        out = [self._get_data(vid_id).T[None, :, :]]
        if  self.return_label:
            out.append(self._strain_dict[strain])
        
        if self.return_snp:
            out.append(self._snps[strain].values)
        
        return out
    
def collate_fn(batch):
    out = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
    return out

if __name__ == '__main__':
    import sys
    from path import _root_dirs
    
    p = 'loc' if sys.platform == 'darwin' else 'ox'
    #p = 'tmp'
    root = _root_dirs[p]
    
    emb_set = 'AE_emb_20180206'
    #emb_set = 'angles'
    fname = root + 'experiments/classify_strains/CeNDR_{}.hdf5'.format(emb_set)
    #fname = '/Users/avelinojaver/Documents/Data/classify_strains/CeNDR_{}.hdf5'.format(emb_set)
    #fname = root + 'CeNDR_{}.hdf5'.format(emb_set)
    
    #%%
    
    gen = SkelTrainer(fname = fname, 
                      is_divergent_set = False, 
                      is_tiny = False,
                        return_label = False, 
                        return_snp = True,
                        unsampled_test = True
                        )
    print([gen._strain_dict[x] for x in _divergent_set])
    
    #%%
    
    loader = DataLoader(gen, 
                        batch_size = 2, 
                        collate_fn = collate_fn,
                        num_workers = 2
                        )
    #%%
    gen.test()
    dat = []
    
    #print(gen.video_info.loc[gen.valid_index, 'strain'].value_counts())
    #for ii, D in enumerate(tqdm.tqdm(gen)):
    #    dat.append(D[1])
        
        
        
    for ii, D in enumerate(gen):    
        print([x.shape for x in D])
        if ii > 10:
            break
        break
    #plt.imshow(D[0][0].T, aspect='auto')
    
 