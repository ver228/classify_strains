#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""

import os
import sys

import numpy  as np
import matplotlib.pylab as plt

import tqdm
import torch


#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, 'src')
sys.path.append(src_dir)
from flow import ROIFlowBatch
from models import VAE, VAELoss
if __name__ == '__main__':
    #%%
    mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    feat_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021_featuresN.hdf5'

    vae = VAE(64)
    criterion = VAELoss()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
    #%%
    #optimizer = torch.optim.SGD(vae.parameters(), lr=1e-1, momentum=0.9)
    n_epochs = 1000
    gen = ROIFlowBatch(mask_file, feat_file, batch_size=32, roi_size=128)
    vae.train()
    
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(gen)
        for X in pbar:            
            output = vae.forward(X)
            loss = criterion(output, X)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()       
            
            dd = 'Epoch {} , loss={}'.format(epoch, loss.data[0])
            pbar.set_description(desc=dd, refresh=False)
    
    #%%
    vae.eval()
    decoded, mu, logvar = vae.forward(X)
    
    x = vae.encoder(X).view(-1, 256)
    z = vae.fc21(x)
    
    z = z
    dd2 = vae.decoder(z.view(*z.size(), 1, 1))
    
    cx = X.data.numpy()
    imgs = decoded.data.numpy()
    dd2 = dd2.data.numpy()
    for img, ori, d2 in zip(imgs,cx, dd2):
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(ori), interpolation=None, cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(img), interpolation=None, cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(d2), interpolation=None, cmap='gray')
        
    