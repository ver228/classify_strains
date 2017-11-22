#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:08:01 2017

@author: ajaver
"""

def _test_read():
    from tierpsy.helper.params import read_microns_per_pixel
    microns_per_pixel = read_microns_per_pixel(feat_file)
    
    with pd.HDFStore(feat_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        
    roi_generator = generateMoviesROI(mask_file, trajectories_data, roi_size=128)
    
    frame_data = next(roi_generator)
    for irow, (img_roi, roi_corner) in frame_data.items():
        img_roi_N = (img_roi.astype(np.float32)-90)/255
        row = trajectories_data.loc[irow]
        plt.figure()
        plt.imshow(img_roi_N, interpolation=None, cmap='gray')
        
        skel_id = int(row['skeleton_id'])
        if skel_id > 0:
            with tables.File(feat_file, 'r') as fid:
                skel = fid.get_node('/coordinates/skeletons')[skel_id]
                skel /= microns_per_pixel
                skel -= roi_corner[ None, :]
                plt.plot(skel[..., 0], skel[..., 1])
                plt.plot(skel[0, 0], skel[0, 1], 'o')