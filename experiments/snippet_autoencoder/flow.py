#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:14:05 2018

@author: ajaver
"""
import glob
import os
import pandas as pd
import random
import tables
import numpy as np
import multiprocessing as mp

import torch
from torch.autograd import Variable

from queue import Queue
from threading import Thread

roi_dir_dflt = '/data/ajaver/CeNDR_ROIs/'


def shift_and_normalize(data):
    '''
    shift worms values by an approximation of the removed background. I used the top95 of the unmasked area. 
    I am assuming region of the background is kept.
    '''
    data_m = data.view(np.ma.MaskedArray)
    data_m.mask = data==0
    if data.ndim == 3:
        sub_d = np.percentile(data, 95, axis=(1,2)) #let's use the 95th as the value of the background
        data_m -= sub_d[:, None, None]
    else:
        sub_d = np.percentile(data, 95)
        data_m -= sub_d
        
    data /= 255
    return data


class FlowSampled():
    def __init__(self, 
                 dataset_info, 
                 n_samples, 
                 batch_size, 
                 snippet_size,
                 is_cuda = False,
                 gpu_id = 0):
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.snippet_size = snippet_size
        self.is_cuda = is_cuda
        self.gpu_id = gpu_id
        self.queue = Queue(batch_size)
        self.thread = None
        self._i_sample = 0
        
        self._init_target_fun()
    
    def _init_target_fun(self):
        def fun():
            '''
            Target function for the queue trend. I put it outside the object because
            sometimes there is a problem to passing data to trends since objects cannot be pickled
            '''
            def _read_snippet(rr):
                #unpack
                fname, ini, fin = rr
                with tables.File(fname, 'r') as fid:
                    snippet = fid.get_node('/mask')[ini:fin]
                    return snippet
                    
            def _get_choice():
                fname, roi_lims = random.choice(self.dataset_info)
                _, r_min, r_max = random.choice(roi_lims)
                
                t_lim = r_max - self.snippet_size
                if t_lim < r_min:
                    raise ValueError
                
                ind = random.randint(r_min, r_max - self.snippet_size)
                
                
                return fname, ind, ind + self.snippet_size
            
            tot = self.n_samples*self.batch_size
            
            nn = 0
            n_bads = 0
            while nn < tot:
                try:
                    rr = _get_choice()
                    snippet = _read_snippet(rr)
                    nn += 1
                    n_bads = 0
                except:
                    #I want to have a way to break if i continue to have problems to read files
                    n_bads += 1
                    if n_bads >= 5:
                        raise
                    else:
                        continue
                    
                self.queue.put(snippet)
            
        self._target_fun = fun
    
    def __iter__(self):
        
        self.thread = Thread(target = self._target_fun)
        self.thread.start()
        
        self._i_sample = 0
        return self
            
    def __next__(self):
        self._i_sample += 1
        if self._i_sample <= self.n_samples:
            batch_d = np.array([self.queue.get() for _ in range(self.batch_size)])
            batch_d = shift_and_normalize(batch_d.astype(np.float32)) + 0.5
            
            #add extra dimension (channel)
            S = torch.from_numpy(batch_d[:, None, ...])
            if self.is_cuda:
                S = S.cuda(self.gpu_id)
            S = Variable(S)
            
            return S
        else:
            assert self.queue.empty()
            raise StopIteration
    
    def __len__(self):
        return self.n_samples



def read_dataset_info(roi_dir = roi_dir_dflt):

    def get_file_info(fname):
        try:
            
            with tables.File(fname, 'r') as fid:
                has_finished = fid.get_node('/mask')._v_attrs['has_finished']
                if has_finished != 1:
                    return
                
            with pd.HDFStore(fname, 'r') as fid:
                trajectories_data = fid['/trajectories_data']
        except (KeyError, OSError, tables.exceptions.HDF5ExtError):
            print('error')
            
            return
        except Exception as e:
            print('Error type:', type(e))
            raise e
            
        gg = trajectories_data.groupby('worm_index_joined')
        roi_lims = gg.agg({'roi_index':['min', 'max']})['roi_index']
        roi_lims.reset_index(level=0, inplace=True)
        
        print(fname)
        return (fname, roi_lims.values.tolist())

    fnames = glob.glob(os.path.join(roi_dir, '**', '*ROIs.hdf5'), recursive=True)
    all_data = [x for x in map(get_file_info, fnames) if x is not None]
    return all_data

#%%
if __name__ == '__main__':
    import pickle
    import tqdm
    import matplotlib.pylab as plt
    #%%
    all_data = read_dataset_info()
    #%%
    all_data = sorted(all_data)
    random.seed(777)
    random.shuffle(all_data)
    
    tot = len(all_data)
    tiny_set = all_data[:5]
    
    ind_p = round(tot*0.8)
    
    train_set = all_data[:ind_p]
    test_set = all_data[ind_p:]
    
    all_sets = {'tiny':tiny_set, 
                'test':test_set,
                'train':train_set}
    
    save_name = os.path.join(roi_dir_dflt, 'set_divisions.p')
    with open(save_name, 'wb') as fid:
        pickle.dump(all_sets, fid)
    
    
    
    
    
    #%%
    
    flow_t = FlowSampled(all_data, 10, 2, 255)
    
    #%%
    for rois in tqdm.tqdm(flow_t):
        pass
    #%%
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(rois[1, 0, 0].data.numpy())
    plt.subplot(1,2,2)
    plt.imshow(rois[1, 0, -1].data.numpy())
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(rois[0, 0, 0].data.numpy())
    plt.subplot(1,2,2)
    plt.imshow(rois[0, 0, -1].data.numpy())