#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:04:37 2018

@author: avelinojaver
"""

from path import _root_dirs
import sys

import pandas as pd
import tables
import numpy as np
import glob
import os
import tqdm
import datetime
import warnings
#%%
def df_to_records(df):
    rec = df.to_records(index=False)
    
    #i want to use this to save into pytables, but pytables does not support numpy objects 
    #so i want to cast the type to the longest string size
    new_dtypes = []
    for name, dtype in rec.dtype.descr:
        if dtype.endswith('O'):
            max_size = max(len(x) for x in rec[name])
            new_dtype = '<S{}'.format(max_size)
            new_dtypes.append((name, new_dtype))
        else:
            new_dtypes.append((name, dtype))
    rec = rec.astype(np.dtype(new_dtypes))
    return rec
#%%
def _h_angles(skeletons):
    '''
    Get skeletons angles
    '''
    dd = np.diff(skeletons, axis=1)
    angles = np.arctan2(dd[..., 0], dd[..., 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1)

    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    return angles, mean_angles


#%%


if __name__ == '__main__':
    p = 'loc' if sys.platform == 'darwin' else 'ox'
    root = _root_dirs[p]
    
    root_dir = root + 'screenings/CeNDR/Results'
    save_file = root + 'experiments/classify_strains/CeNDR_angles.hdf5'
    
    f_ext = '_featuresN.hdf5'
    col_label = 'skeleton_id'
    
    fnames = glob.glob(os.path.join(root_dir, '**', '*' + f_ext), recursive = True)
    fnames = sorted(fnames)
    #%%
    
    video_info = []
    for fname in fnames:
        bn = os.path.basename(fname).replace(f_ext, '')
        parts = bn.split('_')
        strain = parts[0]
        n_worms = int(parts[1][5:])
        
        dt =datetime.datetime.strptime(parts[-2] + parts[-1], '%d%m%Y%H%M%S')
        date_str = datetime.datetime.strftime(dt, '%Y-%m-%d')
        time_str = datetime.datetime.strftime(dt, '%H:%M:%S')
        
        fname_r = fname.replace(root_dir, '')
        set_n = int(fname_r.partition('_Set')[-1][0])
        
        row = (strain, set_n, n_worms, date_str, time_str, fname_r)
        video_info.append(row)
    
    video_info = pd.DataFrame(video_info, columns=['strain', 'set_n', 'n_worms', 'date', 'time', 'file_path'])
    
    
    #%%
    
    traj_ranges = []
    
    filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )
    
    irow = 0
    with tables.File(save_file, 'w') as fid:
        angle_tab = fid.create_earray('/', 
                                      'angles', 
                                      atom = tables.Int32Atom(), 
                                      shape = (0, 48), 
                                      filters  =filters
                                      )
        
        for ifname, fname in tqdm.tqdm(enumerate(fnames), total=len(fnames)):
            with pd.HDFStore(fname, 'r') as fid:
                trajectories_data = fid['/trajectories_data']
            trajectories_data = trajectories_data[['worm_index_joined', 'frame_number', col_label]]
            trajectories_data = trajectories_data[trajectories_data[col_label]>=0]
            
            
            
            with tables.File(fname, 'r') as fid:
                skels_g = fid.get_node('/coordinates/skeletons')
                for w_ind, dat in trajectories_data.groupby('worm_index_joined'):
                    ini, top = dat[col_label].min(), dat[col_label].max()
                    w_skels = skels_g[ini:top]
                    
                    w_skels[np.isnan(w_skels)] = 0
                    angs, _ = _h_angles(w_skels)
                    angle_tab.append(angs)
                    row = (ifname, w_ind,
                           dat['frame_number'].min(), dat['frame_number'].max(),
                           irow, irow + angs.shape[0],
                           )
                    irow += angs.shape[0] + 1
                    
                    traj_ranges.append(row)
                
        #%%
    traj_ranges = pd.DataFrame(traj_ranges, columns=['video_id', 'worm_index', 'frame_ini', 'frame_fin', 'row_ini', 'row_fin'])
    
    #%% save data
    video_info_rec = df_to_records(video_info)
    traj_ranges_rec = df_to_records(traj_ranges)
    
    save_name = os.path.join(root_dir, 'trajectories_ranges.hdf5')
    with tables.File(save_name, 'w') as fid:
        fid.create_table('/', 'trajectories_ranges', obj  =traj_ranges_rec)   
        fid.create_table('/', 'video_info', obj = video_info_rec)