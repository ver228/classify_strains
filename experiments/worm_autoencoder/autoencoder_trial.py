#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import numpy  as np
import matplotlib.pylab as plt
import tqdm


import torch
from torch import nn


import os
import sys
#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, 'src')
sys.path.append(src_dir)
from flow import ROIFlowBatch
from models import AE

mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
feat_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021_featuresN.hdf5'


if __name__ == '__main__':
    ae = AE(64)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-2)
    #%%
    #optimizer = torch.optim.SGD(vae.parameters(), lr=1e-1, momentum=0.9)
    
    n_epochs = 100
    
    #%%
    gen = ROIFlowBatch(mask_file, feat_file, batch_size=32, roi_size=128)
    ae.train()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
        for X in pbar:
            decoded = ae.forward(X)
            loss = criterion(decoded, X)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()       
            
            dd = 'Epoch {} , loss={}'.format(epoch, loss.data[0])
            pbar.set_description(desc=dd, refresh=False)
    #%%
    cx = X.data.numpy()
    decoded = ae.forward(X)
    imgs = decoded.data.numpy()
    for img, ori in zip(imgs,cx):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.squeeze(ori), interpolation=None, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(np.squeeze(img), interpolation=None, cmap='gray')
        
    