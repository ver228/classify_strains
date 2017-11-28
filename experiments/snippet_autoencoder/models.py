#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:08:01 2017

@author: ajaver
"""
import math
from torch import nn

class AE3D(nn.Module):
    def __init__(self, embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 32, 7, padding=3),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2), 
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2)), 
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2)), 
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(), 
            nn.MaxPool3d((4, 2, 2)), #b, 256, 1, 1
        )
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        self.fc_decoder = nn.Linear(self.embedding_size, 256)
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 7),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 32, 3, stride=2),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, 16, 3, stride=2),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16, 16, 3, stride=2),  # b, 1, 127, 127
            nn.LeakyReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=(2,1,1)),  # b, 1, 255, 129, 129
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def encoder(self, x):
         x = self.cnn_encoder(x).view(-1, 256)
         x = self.fc_encoder(x)
         return x
        
    def decoder(self, x):
        x = self.fc_decoder(x)
        x = x.view(-1, 256, 1, 1, 1)
        x = self.cnn_decoder(x)
        x = x[..., :-1, :-1]
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    import os
    from flow import ROIFlowBatch
    
    data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/videos'
    fname = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    mask_file = os.path.join(data_dir,fname)
    feat_file = os.path.join(data_dir,fname.replace('.hdf5', '_featuresN.hdf5'))
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             roi_size = 128,
                             batch_size = 2,
                             snippet_size = 255,
                             is_cuda = False
                             )
    #%%
    for S in generator:
        break
    model = AE3D()
    print(S.size())
    #%%
    x = model.encoder(S)
    x = model.fc_decoder(x)
    x = x.view(-1, 256, 1, 1, 1)
    print(x.size())
    
    xs = model.cnn_decoder(x)
    print(xs.size())
    