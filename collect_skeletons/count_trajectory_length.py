#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:52:44 2017

@author: ajaver
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import tables

import multiprocessing as mp
from tierpsy.helper.params import read_fps


def read_traj_l(dd):
    irow, row = dd
    print(irow+1, len(experiments_df))
    features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
    
    with pd.HDFStore(features_file, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        return timeseries_data['worm_index'].value_counts()


if __name__ == '__main__':
    sys.path.append('/Users/ajaver/Documents/GitHub/process-rig-data/process_files')
    from misc import get_rig_experiments_df

    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')

    set_type = 'featuresN'
    
    save_dir = './results_{}'.format(set_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments_df = get_rig_experiments_df(features_files, csv_files)
    experiments_df = experiments_df.sort_values(by='video_timestamp').reset_index()  
    experiments_df['id'] = experiments_df.index
    
    experiments_df = experiments_df[['id', 'strain', 'directory', 'base_name', 'exp_name']]
    experiments_df['fps'] = np.nan
    
    p = mp.Pool(20)
    
    all_traj = p.map(read_traj_l, experiments_df.iterrows())
    
        #%%   
    import matplotlib.pylab as plt
    V = np.concatenate([x.values for x in all_traj])
    plt.figure(figsize=(10,5))
    plt.hist(np.log10(V/25), 100)
    yy = plt.ylim()
    plt.plot(np.log10((90, 90)), yy, '--k', lw=3)
    plt.plot(np.log10((10, 10)), yy, '--k', lw=3)
    
    plt.xlabel(r'Track length $\log_{10}(\mathrm{seconds})$')
    plt.ylabel('Counts')
    plt.savefig('/Users/ajaver/counts.png')
    #%%
    cc, edges = np.histogram(V, 10000)
    bins = (edges[1:] + edges[:-1])/2
    #%%
    tot_frames = cc*bins
    plt.figure(figsize=(10,5))
    
    yy = np.cumsum(tot_frames[::-1])[::-1]
    yy = yy/yy[0]
    
    plt.plot(bins/25, yy)
    plt.xscale('log')
    yy = plt.ylim()
    plt.plot((90, 90), yy, '--k', lw=3)
    plt.plot((10, 10), yy, '--k', lw=3)
    plt.xlabel('Track length [seconds]')
    plt.ylabel('Cumulative fraction of skeletons')
    plt.savefig('/Users/ajaver/counts_cumulative.png')
    #%%
    #plt.figure(figsize=(10,5))
    #plt.hist(np.log10(V/25), 100, normed=1, cumulative=-1)
    #yy = plt.ylim()
    #plt.plot(np.log10((90, 90)), yy, '--k', lw=3)
    #plt.plot(np.log10((10, 10)), yy, '--k', lw=3)