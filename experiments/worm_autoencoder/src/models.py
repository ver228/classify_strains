#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:08:01 2017

@author: ajaver
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, output, target):
        decoded, mu, logvar = output
        BCE = F.mse_loss(decoded, target)
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= decoded.view(-1).size(0)
        return BCE + KLD


class VAE(nn.Module):
    def __init__(self, embedding_size=128):
        self.embedding_size = embedding_size
        super().__init__()
        self.cnn_encoder = nn.Sequential(
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

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1),  # b, 1, 128, 128
            nn.Tanh()
        )
        
        self.fc_encoder_mu = nn.Linear(256, self.embedding_size)
        self.fc_encoder_logvar = nn.Linear(256, self.embedding_size)
        
        self.fc_decoder = nn.Linear(self.embedding_size, 256)
        
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
    
    def encoder(self, x_in):
        x = self.cnn_encoder(x_in).view(-1, 256)
        mu = self.fc_encoder_mu(x)
        logvar = self.fc_encoder_logvar(x)
        return mu, logvar
    
    def decoder(self, z):
        x = self.fc_decoder(z)
        x = self.cnn_decoder(x.view(*x.size(), 1, 1))
        return x
    
    def forward(self, x_in):
        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)
        x_out = self.decoder(z)
        return x_out, z, logvar

class AE(nn.Module):
    def __init__(self, embedding_size = 256):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.cnn_encoder = nn.Sequential(
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
        self.fc_encoder = nn.Linear(256, self.embedding_size)
        self.fc_decoder = nn.Linear(self.embedding_size, 256)
        
        self.cnn_decoder = nn.Sequential(
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
        x = x.view(-1, 256, 1, 1)
        x = self.cnn_decoder(x)
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x