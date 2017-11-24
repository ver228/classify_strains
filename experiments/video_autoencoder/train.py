#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:28:35 2017

@author: ajaver
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from flow import ROIFlowBase, shift_and_normalize
from models import AE3D
if __name__ == '__main__':
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    feat_file = mask_file.replace('.hdf5', '_featuresN.hdf5')
        
    g = ROIFlowBase(mask_file, feat_file)
    
    model = AE3D()
    #%%
    snippet_size = 128
    
    rois_collected = {}
    for frame_number in range(snippet_size):
        frame_rois = g._get_frame_rois(frame_number)
        for w, (roi, _) in frame_rois.items():
            if not w in rois_collected:
                rois_collected[w] = []
            rois_collected[w].append(roi[None, ...])
    
    all_snippets = []
    for w, rois_l in rois_collected.items():
        if len(rois_l) < snippet_size:
            continue
        
        snippet = np.concatenate(rois_l[:snippet_size])
        snippet = shift_and_normalize(snippet.astype(np.float32)) + 0.5
        
        all_snippets.append(snippet[None, None, ...])
    #%%
    batch_size = 4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    import tqdm
    for epoch in range(1000):
        pbar = tqdm.tqdm(range(0, len(all_snippets), batch_size))
        for ii in pbar:
            snippets = all_snippets[ii:ii+batch_size]
            S = np.concatenate(snippets)
            S = torch.from_numpy(S)
            S = Variable(S)
            
            output = model(S)
            loss = criterion(output, S)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description('Epoch {} loss {}'.format(epoch, loss.data[0]), refresh=False)
                