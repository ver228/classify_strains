#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:58:43 2017

@author: ajaver
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:24:09 2017

@author: ajaver
"""
import tables
import pandas as pd
import random
import numpy as np

import torch
from torch.autograd import Variable

from scipy.ndimage.interpolation import rotate

def getWormROI(img, CMx, CMy, roi_size=128):
    '''
    Extract a square Region Of Interest (ROI)
    img - 2D numpy array containing the data to be extracted
    CMx, CMy - coordinates of the center of the ROI
    roi_size - side size in pixels of the ROI

    -> Used by trajectories2Skeletons
    '''

    if np.isnan(CMx) or np.isnan(CMy):
        return np.zeros(0, dtype=np.uint8), np.array([np.nan] * 2)

    roi_center = int(roi_size) // 2
    roi_range = np.round(np.array([-roi_center, roi_center]))

    # obtain bounding box from the trajectories
    range_x = (CMx + roi_range).astype(np.int)
    range_y = (CMy + roi_range).astype(np.int)

    if range_x[0] < 0:
        range_x[0] = 0
    if range_y[0] < 0:
        range_y[0] = 0
    
    if range_x[1] > img.shape[1]:
        range_x[1] = img.shape[1]
    if range_y[1] > img.shape[0]:
        range_y[1] = img.shape[0]

    worm_img = img[range_y[0]:range_y[1], range_x[0]:range_x[1]]

    roi_corner = np.array([range_x[0], range_y[0]])

    return worm_img, roi_corner

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


if __name__ == '__main__':
    import time
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
    
    with pd.HDFStore(feat_file) as fid:
        trajectories_data = fid['trajectories_data']
    group_by_frames = trajectories_data.groupby('frame_number')
    group_by_index = trajectories_data.groupby('worm_index_joined')
    
    
    fid_masks = tables.File(mask_file)
    masks = fid_masks.get_node('/mask')    

    n_batch = 128
    tic = time.time()
    worms_in_frame = {}
    for tt in range(2*n_batch):
        frame_img = masks[tt]
        frame_data = group_by_frames.get_group(tt)
        for irow, row in frame_data.iterrows():
            roi_img, roi_corner = getWormROI(frame_img, 
                                  row['coord_x'], 
                                  row['coord_y'], 
                                  roi_size = 128
                                  )
            w_i = int(row['worm_index_joined'])
            worms_in_frame[w_i] = (roi_img, roi_corner)
    
    all_snippets = []
    for w, rois_l in worms_in_frame.items():
        if len(rois_l) < 128:
            continue
        
        snippet = np.concatenate(rois_l[:128])
        snippet = shift_and_normalize(snippet.astype(np.float32)) + 0.5
        
        all_snippets.append(snippet[None, None, ...])
        
    print('S', time.time()-tic)
    #%%
    
    
    #%%
    n_batch=128
    tic = time.time()
    for tt in range(0, 2*n_batch, n_batch):
        img = masks[tt: tt + n_batch]
    print('B', time.time()-tic)
    
    
    
    #%%

#    g = ROIFlowBase(mask_file, feat_file)
#    
#    #%%
#    snippet_size = 25
#    
#    rois_collected = {}
#    for frame_number in range(snippet_size):
#        frame_rois = g._get_frame_rois(frame_number)
#        for w, (roi, _) in frame_rois.items():
#            if not w in rois_collected:
#                rois_collected[w] = []
#            rois_collected[w].append(roi[None, None, ...])
#    
#    all_snippets = []
#    for w, rois_l in rois_collected.items():
#        snippet = np.concatenate(rois_l)
#        all_snippets.append(snippet[None, ...])
    
