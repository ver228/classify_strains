#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:01:29 2018

@author: avelinojaver
"""
import os
import itertools
import pickle
import numpy as np
import pandas as pd


root_dir = os.path.dirname(__file__)
DFLT_SNP_FILE = os.path.join(root_dir, 'CeNDR_snps.csv')
DFLT_FOLDS_FILE = os.path.join(root_dir, 'fold_dict.p')

def read_CeNDR_snps(source_file = DFLT_SNP_FILE):
    snps = pd.read_csv(source_file)
    
    info_cols = snps.columns[:4]
    strain_cols = snps.columns[4:]
    snps_vec = snps[strain_cols].copy()
    snps_vec[snps_vec.isnull()] = 0
    snps_vec = snps_vec.astype(np.int8)
    
    
    snps_c = snps[info_cols].join(snps_vec)
    
    r_dtype = []
    for col in snps_c:
        dat = snps_c[col]
        if dat.dtype == np.dtype('O'):
            n_s = dat.str.len().max()
            dt = np.dtype('S%i' % n_s)
        else:
            dt = dat.dtype
        r_dtype.append((col, dt))
    
    snps_r = snps_c.to_records(index=False).astype(r_dtype)
    snps_r = pd.DataFrame(snps)
    
    return snps_r

def get_strains_ids(snps):
    valid_strains = snps.columns[4:].tolist()
    strain_dict = {k:ii for ii, k in enumerate(valid_strains)}
    return strain_dict


def add_bn_series(df):
    func = lambda x : os.path.basename(x).rpartition('_')[0]
    df['base_name'] = df['file_path'].apply(func)
    return df


def save_folds_dict(df, save_file=DFLT_FOLDS_FILE, seed = 777):
    df = add_bn_series(df)
    folds_dict = {}
    for ss, dat in df.groupby('strain'):
        gen = itertools.cycle(range(3))
        for bn in dat['base_name'].values:
            folds_dict[bn] = next(gen)
    
    with open(save_file, 'wb') as fid:
        pickle.dump(folds_dict, fid)

def add_folds(df):
    df = add_bn_series(df)
    with open(DFLT_FOLDS_FILE, 'rb') as fid:
        folds_dict = pickle.load(fid)
    df['fold'] = df['base_name'].map(folds_dict)
    return df
    

#%%
if __name__ == '__main__':
    
    #%%
    fname = '/Users/avelinojaver/Documents/Data/experiments/classify_strains/CeNDR_angles.hdf5'
    
    with pd.HDFStore(fname) as fid:
        video_info = fid['/video_info']
        video_info['strain'] = video_info['strain'].str.strip(' ')
        
        
    video_info = add_folds(video_info)
    #%%
    
    
    