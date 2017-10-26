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

from tierpsy.helper.params import read_fps
from collect_helper import collect_skeletons, add_sets_index, TABLE_FILTERS

def ini_experiments_df():

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
    print('Reading fps...')
    for irow, row in experiments_df.iterrows():
        print(irow+1, len(experiments_df))
        features_file = os.path.join(row['directory'], row['base_name'] + '_featuresN.hdf5')
        fps = read_fps(features_file)
        experiments_df.loc[irow, 'fps'] = fps
    
    return experiments_df

def read_CeNDR_snps():
    fname = '/Users/ajaver/Documents/GitHub/process-rig-data/tests/CeNDR/CeNDR_snps.csv'

    snps = pd.read_csv(fname)
    
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
    return snps_r

if __name__ == '__main__':
    #%%
    main_file = '/Users/ajaver/Desktop/CeNDR_skel_smoothed1.hdf5'
    
    experiments_df = ini_experiments_df()

    collect_skeletons(experiments_df, 
                      main_file,  
                      file_ext = '_featuresN.hdf5'
                      )
            
    
    #%%
    #get snps vector
    snps = read_CeNDR_snps()
    with tables.File(main_file, 'r+') as fid:
        if '/snps_data' in fid:
            fid.remove_node('/snps_data')
        fid.create_table(
            '/',
            'snps_data',
            obj = snps,
            filters = TABLE_FILTERS
            )
    #%%
    add_sets_index(main_file, val_frac = 0.1, test_frac = 0.1)