#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:49:50 2017

@author: ajaver
"""
import os
import torch
import numpy as np
import matplotlib.pylab as plt
from train import AE, VAE, is_cuda, feat_file, mask_file, ROIFlowBatch

model_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/worm_autoencoder'

#model_path = os.path.join(model_dir_root, 'AE__20171123_000043', 'checkpoint.pth.tar')
#model = AE()

model_path = os.path.join(model_dir_root, 'VAE__20171123_005219', 'checkpoint.pth.tar')
model = VAE()

checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

if __name__ == '__main__':
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             is_cuda = is_cuda,
                             batch_size = 32, 
                             roi_size = 128,
                             is_shuffle = True
                             )
    
    
    for X in generator:
        break
    #decoded = model.forward(X)
    decoded = model.forward(X)[0]
    
    cx = X.data.squeeze().numpy()
    imgs = decoded.data.squeeze().numpy()
    for img, ori in zip(imgs,cx):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(ori, interpolation=None, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(img, interpolation=None, cmap='gray')
        