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

class ROIFlowBase():
    _current_frame = -1
    _frames2iter = []
    _size = None
    def __init__(self, 
                 mask_file, 
                 feat_file, 
                 roi_size=128,
                 is_shuffle = False):
        self.mask_file = mask_file
        self.feat_file = feat_file
        self.roi_size = roi_size
        self.is_shuffle = is_shuffle
        
        with pd.HDFStore(self.feat_file) as fid:
            trajectories_data = fid['trajectories_data']
        
        self.trajectories_data = trajectories_data
        self.group_by_frames = trajectories_data.groupby('frame_number')
        self.group_by_index = trajectories_data.groupby('worm_index_joined')
        
        
        self.fid_masks = tables.File(self.mask_file)
        self.masks = self.fid_masks.get_node('/mask')
        
        self.__iter__() #init iteration in case somebody call __next__ 
        
    def __iter__(self):
        self._frames2iter = list(self.group_by_frames.groups.keys())
        if self.is_shuffle:
            random.shuffle(self._frames2iter)
        
        return self
    
    def __next__(self):
        try:
            
            self._current_frame = self._frames2iter.pop(0)
            return self._get_frame_rois(self._current_frame)
        except IndexError:
            raise StopIteration
    
    
    def _get_frame_rois(self, frame_number):
        frame_data = self.group_by_frames.get_group(frame_number)
        frame_img = self.masks[frame_number]
        worms_in_frame = {}
        
        for irow, row in frame_data.iterrows():
            roi_img, roi_corner = getWormROI(frame_img, 
                                  row['coord_x'], 
                                  row['coord_y'], 
                                  self.roi_size
                                  )
        
            
            if any(x != self.roi_size for x in roi_img.shape):
                continue
            
            w_i = int(row['worm_index_joined'])
            worms_in_frame[w_i] = (roi_img, roi_corner)
            
        return worms_in_frame
        
    def _transform(self, img):
        #random rotation
        if self.training:
            ang = random.uniform(-180, 180)
        return rotate(img, ang, reshape=False)

    def __len__(self):
        if self._size is None:
            self._size = len(self.trajectories_data.groups.keys())
        return self._size
        

class ROIFlowBatch(ROIFlowBase):
    def __init__(self, 
                 mask_file, 
                 feat_file, 
                 batch_size = 32,
                 is_torch = True,
                 **argkws):
        super().__init__(mask_file, feat_file, **argkws)
        self.batch_size = batch_size
        self.is_torch = is_torch
        
    def __iter__(self):
        #initialize iterator by frames
        super().__iter__()
        
        
        remainder = []
        while True:
            frame_data = super().__next__()
           
            remainder += [x[None, ... ] for x, _ in frame_data.values()]
            
            if len(remainder) >= self.batch_size:
                chunk = np.concatenate(remainder[:self.batch_size])
                remainder = remainder[self.batch_size:] 
                
                if self.is_torch:
                    chunk = self._to_torch(chunk)
                
                yield chunk
    
    def _to_torch(self, dat):
        dat = shift_and_normalize(dat.astype(np.float32)) + 0.5
        dat = dat[:, None, ...]
        X = torch.from_numpy(dat)
        X = Variable(X)
        return X
    
    def __len__(self):
        if self._size is None:
            self._size = self.trajectories_data.shape[0]//self.batch_size
        return self._size

if __name__ == '__main__':
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
        
    g = ROIFlowBatch(mask_file, feat_file)
    
    for dat in g:
        print(dat.size())
            
    
    
    