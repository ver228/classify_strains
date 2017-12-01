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

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, output, target):
        decoded, mu, logvar = output
        BCE = F.binary_cross_entropy(decoded, target)
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= decoded.view(-1).size(0)
        return BCE + KLD


class SNPMapperVAE(nn.Module):
    def __init__(self, embedding_size=64, snps_size=37851):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.snps_size = snps_size
        
        self.fc_encoder_1 = nn.Linear(snps_size, 2048)
        self.fc_encoder_mu = nn.Linear(2048, embedding_size)
        self.fc_encoder_logvar = nn.Linear(2048, embedding_size)
        
        self.decoder = nn.Sequential(
                nn.Linear(embedding_size, 2048),
                nn.Linear(2048, snps_size),
                nn.Tanh()
                )
        
        
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu
    
    def encoder(self, x_in):
        x = self.fc_encoder_1(x_in)
        mu = self.fc_encoder_mu(x)
        logvar = self.fc_encoder_logvar(x)
        return mu, logvar
    
    
    def forward(self, x_in):
        mu, logvar = self.encoder(x_in)
        z = self.reparameterize(mu, logvar)
        
        x = self.decoder(z)
        return x, mu, logvar

