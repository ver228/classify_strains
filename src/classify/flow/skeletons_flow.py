#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:20:28 2017
@author: ajaver

#add modifications by Andreas

"""
import random
import time
import numpy as np
import pandas as pd
import tables
import torch
import math
from .skeletons_transform import get_skeleton_transform, check_valid_transform

IS_CUDA = torch.cuda.is_available()

class SkeletonsFlowBase():
    
    _n_classes = None
    _n_skeletons = None
    _valid_strains = None
    
    def __init__(self,
                 n_batch = 1, 
                 data_file = '',
                 set_type = None,
                 min_num_experiments = 1,
                 valid_strains = None,
                 sample_size_seconds = 90.,
                 sample_frequency_s =1 / 10.,
                 transform_type = 'angles',
                 is_normalized = False,
                 is_torch = False
                 ):

        check_valid_transform(transform_type)
        
        self.n_batch = n_batch
        self.sample_size_seconds = sample_size_seconds
        self.sample_frequency_s = sample_frequency_s
        self.sample_size = int(round(sample_size_seconds / sample_frequency_s))
        self.data_file = data_file
        self.transform_type = transform_type
        self.is_normalized = is_normalized
        self.is_torch = is_torch
        
        with pd.HDFStore(self.data_file, 'r') as fid:
            skeletons_ranges = fid['/skeletons_groups']
            experiments_data = fid['/experiments_data']
            self.strain_codes = fid['/strains_codes']
            self.strain_codes.index = self.strain_codes['strain_id']

            # read SNP only valid in CeNDR
            if '/snps_data' in fid:
                self.snps_data = fid['/snps_data']

        # Join the experiments and skeletons groups tables
        # I must use pd.join  NOT pd.merge to keep the same indexes as skeletons groups
        # otherwise the '/index_groups' subdivision will break
        cols_to_use = experiments_data.columns.difference(
            skeletons_ranges.columns)
        experiments_data = experiments_data[cols_to_use]
        experiments_data = experiments_data.rename(
            columns={'id': 'experiment_id'})
        skeletons_ranges = skeletons_ranges.join(
            experiments_data.set_index('experiment_id'), on='experiment_id')
        skeletons_ranges = skeletons_ranges.join(
            self.strain_codes.set_index('strain'), on='strain')
        
        self.data = tables.File(self.data_file, 'r')

        if set_type is not None:
            assert set_type in ['train', 'test', 'val']
            # use previously calculated indexes to divide data in training, validation and test sets
            valid_indices = self.data.get_node('/index_groups/' + set_type)[:]
            skeletons_ranges = skeletons_ranges.loc[valid_indices]

        # filter data to contain only the valid strains given
        if valid_strains is not None:
            skeletons_ranges = skeletons_ranges[
                skeletons_ranges['strain'].isin(valid_strains)]

        # minimum number of experiments/videos per strain
        skeletons_ranges = skeletons_ranges.groupby('strain_id').filter(
            lambda x: len(x['experiment_id'].unique()) >= min_num_experiments)

        self.skeletons_ranges = skeletons_ranges
    
    def _transform(self, skeletons):
        return get_skeleton_transform(skeletons, 
                               transform_type = self.transform_type,
                               is_normalized=self.is_normalized)
    
    def _read_skeletons(self, ini_r, fin_r, fps):
        # TODO: might be even faster to
        while True:
            try:
                # read data. I use a while to protect from fails of data
                skeletons = self.data.get_node('/skeletons_data')[
                            ini_r:fin_r + 1, :, :]
                break
            except KeyError:
                print(
                    'There was an error reading the file, I will try again...')
                time.sleep(1)
            
        
        # resample the skeletons to match the expected sample_frequency
        tot = np.round(skeletons.shape[0]/(self.sample_frequency_s*fps))
        
        #print(tot, skeletons.shape[0])
        ind_s = np.linspace(0, skeletons.shape[0]-1, tot)
        ind_s = np.round(ind_s).astype(np.int32)
        
        skeletons = skeletons[ind_s]
        return skeletons
    
    def _to_torch(self, X, Y):
        X = np.rollaxis(X, -1, 1) # the channel dimension must be the second one
        
        Xt = torch.from_numpy(X).float()
        Yt = torch.from_numpy(Y).long()
        
        if IS_CUDA:
            Xt = Xt.cuda()
            Yt = Yt.cuda()
            
        input_var = torch.autograd.Variable(Xt)
        target_var = torch.autograd.Variable(Yt)
        
        return input_var, target_var
    
    def _serve_chunk(self, chunks):
        X, Y =  map(np.array, zip(*chunks))
        if self.is_torch:
            X, Y  = self._to_torch(X, Y )
        
        return X, Y 
    

    @property
    def n_segments(self):
        return self.skeletons_ranges.shape[0]

    @property
    def n_skeletons(self):
        if self._n_skeletons is None:
            dd = self.skeletons_ranges['fin'] - self.skeletons_ranges['ini'] + 1
            self._n_skeletons = dd.sum()
        return self._n_skeletons

    @property
    def n_classes(self):
        # number of classes for the one-hot encoding
        if self._n_classes is None:
            self._n_classes = int(self.strain_codes['strain_id'].max() + 1)
        return self._n_classes
    
    @property
    def n_channels(self):
        if self.transform_type == 'xy':
            return 2
        else:
            return 1

    @property
    def valid_strains(self):
        if self._valid_strains is None:
            self._valid_strains = list(self.skeletons_ranges['strain'].unique())
        return self._valid_strains
    


class SkeletonsFlowShuffled(SkeletonsFlowBase):
    _epoch_size = None
    _i_epoch = 0
    
    def __init__(self, **argkws):
        super().__init__(**argkws)
        
        
        # Only used when suffle == False.
        self.skeleton_id = -1
        
        # filter the chucks of continous skeletons to have at least the required sample size
        good = self.skeletons_ranges.apply(lambda x: x['fps'] * (
            x['fin'] - x['ini']) >= self.sample_size_seconds, axis=1)
        self.skeletons_ranges = self.skeletons_ranges[good]
        self.skel_range_grouped = self.skeletons_ranges.groupby('strain_id')
        self.strain_ids = list(map(int, self.skel_range_grouped.indices.keys()))
    
    def __iter__(self):
        self._i_epoch = 0
        return self
    
    def __next__(self):
        if self._i_epoch >= self.epoch_size:
            raise StopIteration()
        
        self._i_epoch += 1
        
        chunks = [self._next_single() for n in range(self.n_batch)]
        skeletons, strain_ids = self._serve_chunk(chunks)
        return skeletons, strain_ids
    
    
    
    def _random_choice(self):
        strain_id, = random.sample(self.strain_ids, 1)
        gg = self.skel_range_grouped.get_group(strain_id)
        ind, = random.sample(list(gg.index), 1)
        skeletons = self.prepare_skeleton(ind)

        return strain_id, skeletons
    
    def _random_transform(self, skeletons):
        # random rotation on the case of skeletons
        theta = random.uniform(-np.pi, np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

        skel_r = skeletons.copy()
        for ii in range(skel_r.shape[1]):
            skel_r[:, ii, :] = np.dot(rot_matrix, skeletons[:, ii, :].T).T

        # random mirrowing. It might be a problem since the skeletons do have left right orientation
        # for ii in range(skel_r.shape[-1]):
        #    skel_r[:, :, ii] *= random.choice([-1, 1])

        return skel_r
    
    def prepare_skeleton(self, ind):
        dat = self.skeletons_ranges.loc[ind]
        fps = dat['fps']

        # randomize the start
        sample_size_frames = int(round(self.sample_size_seconds * fps))
        r_f = dat['fin'] - sample_size_frames
        ini_r = random.randint(dat['ini'], r_f)
        fin_r = ini_r + sample_size_frames
        
        #read skeletons in blocks
        skeletons = self._read_skeletons(ini_r, fin_r, fps)
        skeletons = skeletons[:self.sample_size]
        assert skeletons.shape[0] == self.sample_size
        
        skeletons = self._transform(skeletons)
        return skeletons
    
    def _next_single(self):
        strain_id, skeletons = self._random_choice()
        if self.transform_type == 'xy':
            skeletons = self._random_transform(skeletons)
            
        strain_id, skeletons = self._random_choice()
        if self.transform_type == 'xy':
            skeletons = self._random_transform(skeletons)
        return skeletons, strain_id

    @property
    def epoch_size(self):
        #get an estimated epoch size in relation with the number of skeletons and the movies fps
        if self._epoch_size is None:
            fps = self.skeletons_ranges['fps'].median()
            expected_sample_size = self.sample_size*(fps*self.sample_frequency_s)
            self._epoch_size = self.n_skeletons/expected_sample_size/self.n_batch
            self._epoch_size = int(math.ceil(self._epoch_size))
        return self._epoch_size
    
    
    def __len__(self):
        return self.epoch_size

    
class SkeletonsFlowFull(SkeletonsFlowBase):
    _rows2iter = None
    _total = None
    def __init__(self, gap_btw_samples_s = None,  **argkws):
        super().__init__(**argkws)
        self.skeleton_id = -1
        
        if gap_btw_samples_s is None:
            gap_btw_samples_s = self.sample_size_seconds/2
        self.gap_btw_samples_s = gap_btw_samples_s
        self.gap_btw_samples =  int(round(gap_btw_samples_s/self.sample_frequency_s))
        
    def _prepare_chunks(self, row):
        strain_id = row['strain_id']
        
        skeletons = self._read_skeletons(row['ini'], row['fin'], row['fps'])
        skeletons_t = self._transform(skeletons)
        
        fin = skeletons.shape[0] - self.sample_size
        chunks = [(skeletons_t[tt: tt + self.sample_size], strain_id)
            for tt in range(0, fin, self.gap_btw_samples)]
        
        return chunks
    
    def __iter__(self):
        remainder = []
        for irow, row in self.skeletons_ranges.iterrows():
            chunks = self._prepare_chunks(row)
            remainder = remainder + chunks
            while len(remainder) >= self.n_batch:
                chunks = remainder[:self.n_batch]
                remainder = remainder[self.n_batch:]
                yield self._serve_chunk(chunks)
        
        if remainder:
            yield self._serve_chunk(remainder)
      
    def __len__(self):
        if self._total is None:
            tot = 0
            for irow, row in self.skeletons_ranges.iterrows():
                t_size = row['fin'] - row['ini'] + 1 
                # resample the skeletons to match the expected sample_frequency
                n_skels = np.round(t_size/(self.sample_frequency_s*row['fps']))
                
                fin = n_skels - self.sample_size
                tot += math.ceil(fin/self.gap_btw_samples)
            
            self._total = int(math.ceil((tot/self.n_batch)))
            
        return self._total