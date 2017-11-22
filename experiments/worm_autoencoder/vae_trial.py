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
import math
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, 'src')
sys.path.append(src_dir)
from flow import ROIFlowBatch



def loss_function(target, decoded, mu, logvar):
    BCE = F.binary_cross_entropy(target, decoded)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= decoded.view(-1).size(0)

    return BCE + KLD


class VAE(nn.Module):
    def __init__(self):
        self.encoder_layer = 128
        
        
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
            nn.ConvTranspose2d(self.encoder_layer, 128, 7),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1),  # b, 1, 128, 128
            nn.Sigmoid()
            #nn.Tanh()
        )
        
        self.fc21 = nn.Linear(256, self.encoder_layer)
        self.fc22 = nn.Linear(256, self.encoder_layer)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu
    
    def forward(self, x_in):
        x = self.encoder(x_in).view(-1, 256)
        mu = self.fc21(x)
        logvar = self.fc22(x)
        
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z.view(*z.size(), 1, 1))
        return x, mu, logvar



if __name__ == '__main__':
    #%%
    mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    feat_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/Results/CeNDR_Set1_160517/BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021_featuresN.hdf5'

    vae = VAE()
    criterion = loss_function
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
            
            decoded, mu, logvar = vae.forward(X)
            loss = criterion(decoded, X, mu, logvar)      # mean square error
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
        
    