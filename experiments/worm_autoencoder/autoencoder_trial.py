#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
import pandas as pd
import numpy  as np
import matplotlib.pylab as plt
import tqdm


import torch
import math
from torch import nn

mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
feat_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021_featuresN.hdf5'

from roi_flow import ROIFlowBatch


class AE(nn.Module):
    def __init__(self):
        #self.encoder_layer = 128
        
        
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2), #b, 256, 1, 1
        )
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1),  # b, 1, 128, 128
            nn.Sigmoid(),
            nn.Tanh()
        )
        
        #self.fc21 = nn.Linear(256, self.encoder_layer)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    ae = AE()
    
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
        
    