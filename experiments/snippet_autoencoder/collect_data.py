#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:29:49 2018

@author: ajaver
"""

from tierpsy.analysis.ske_create.helperIterROI import generateMoviesROI, pad_if_necessary
import glob
import os
import tables
import pandas as pd
import numpy as np
import multiprocessing as mp


TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='zlib',
    shuffle=True,
    fletcher32=True)


if __name__ == '__main__':
    root_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    mask_dir = os.path.join(root_dir, 'MaskedVideos')
    results_dir = os.path.join(root_dir, 'Results')
    #roi_dir = os.path.join(root_dir, 'ROIs')
    roi_dir = '/data/ajaver/CeNDR_ROIs/'
    
    traj_ext = '_featuresN.hdf5'
    #traj_ext = '_skeletons.hdf5'
    
    #masked_files = glob.glob(os.path.join(maked_files_dir, '**', '*.hdf5',), recursive = True)
    skeletons_files = glob.glob(os.path.join(results_dir, '**', '*' + traj_ext), recursive = True)
    skeletons_files = sorted(skeletons_files)
    fnames = [(ii, ff) for ii,ff in enumerate(skeletons_files)]
    
    roi_size = 128
    min_num_frames = 250
    
    traj_data_dtypes = np.dtype([('frame_number', np.int32),
                                 ('worm_index_joined', np.int32),
                                 ('roi_index', np.int32)
                                 ])
    valid_columns = list(traj_data_dtypes.names)
    
    def _process_file(row):
        experiment_id, skel_file = row
        mask_file = skel_file.replace(results_dir, mask_dir).replace(traj_ext, '.hdf5')
        
        with pd.HDFStore(skel_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        
        trajectories_data['roi_index'] = np.float32(-1)
        
        #reduce the table to save trajectories larger than worm_index_joined
        traj_sizes = trajectories_data['worm_index_joined'].value_counts() 
        valid_ind = traj_sizes[traj_sizes > min_num_frames].index
        good = trajectories_data['worm_index_joined'].isin(valid_ind)
        trajectories_data = trajectories_data[good]
        
        #lets sort by worm_index, and timeframe so the ROIs are stored in a friendlier way
        trajectories_data_r = trajectories_data[valid_columns]
        trajectories_data_r = trajectories_data_r.sort_values(by = ['worm_index_joined', 'frame_number'])
        
        tot_rows = len(trajectories_data_r)
        trajectories_data_r['roi_index'] = np.arange(tot_rows, dtype=np.int32)
        
        #make file to save video
        
        roi_file = mask_file.replace(mask_dir, roi_dir).replace('.hdf5', '_ROIs.hdf5')
        dname = os.path.dirname(roi_file)
        if not os.path.exists(dname):
            os.makedirs(dname)
        
        if os.path.exists(roi_file):
            try:
                with tables.File(roi_file, 'r') as fid:
                    has_finished = fid.get_node('/mask')._v_attrs['has_finished']
                    if has_finished == 1:
                        return
            except (OSError, tables.exceptions.HDF5ExtError, KeyError):
                pass
            except Exception as e:
                print(type(e))
                raise e
        
        with tables.File(roi_file, 'w') as fid_roi:
            rois_c = fid_roi.create_carray('/', 
                            'mask',
                            atom = tables.UInt8Atom(),
                            shape = (tot_rows, roi_size, roi_size),
                            chunkshape = (25, roi_size, roi_size),
                            filters = TABLE_FILTERS
                            )
            rois_c._v_attrs['has_finished'] = 0
            fid_roi.create_table('/',
                        "trajectories_data",
                        trajectories_data_r.to_records(index=False),
                        filters = TABLE_FILTERS)
            
            
            progress_prefix = '{} of {} | {}'.format(experiment_id + 1, len(fnames), os.path.basename(mask_file))
            
            gen = generateMoviesROI(mask_file, trajectories_data, roi_size, progress_prefix = progress_prefix)
            for worms_in_frame in gen:
                for row_ind, (roi_img, roi_corner) in worms_in_frame.items():
                    roi_img, roi_corner = pad_if_necessary(roi_img, roi_corner, roi_size)
                    roi_index = trajectories_data_r.loc[row_ind, 'roi_index']
                    rois_c[roi_index] = roi_img
            
            rois_c._v_attrs['has_finished'] = 1
            
    p = mp.Pool(15)
    list(p.map(_process_file, fnames))
    
    