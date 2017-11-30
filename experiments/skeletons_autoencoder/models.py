#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:55:57 2017

@author: ajaver
"""
import torch
from torch import nn
import torch.nn.functional as F

def weights_init_xavier(m):
    '''
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.uniform(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


class FullAELoss(nn.Module):
    def __init__(self,
                 classification_loss_mixture=1.,
                 embedding_loss_mixture=0.1, 
                 autoencoder_loss_mixture=1.):
        
        super().__init__()
        self.classification_loss_mixture = classification_loss_mixture
        self.embedding_loss_mixture = embedding_loss_mixture
        self.autoencoder_loss_mixture = autoencoder_loss_mixture
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.autoencoder_loss = nn.MSELoss()
        self.embedding_loss = F.mse_loss
        
    def forward(self, output_v, input_v):
        classification, video_embedding, snps_embedding, video_decoded = output_v
        target_classes, video_original = input_v
        
        loss = 0
        if self.classification_loss_mixture > 0:
            clf_loss = self.classification_loss(classification, target_classes)
            loss += clf_loss*self.classification_loss_mixture
        
        if self.embedding_loss_mixture > 0:
            emb_loss = self.embedding_loss(snps_embedding, video_embedding)
            loss += emb_loss*self.embedding_loss_mixture
        
        if self.autoencoder_loss_mixture > 0:
            ae_loss = self.autoencoder_loss(video_decoded, video_original)
            loss += ae_loss*self.autoencoder_loss_mixture
        
        return loss


class CNNEncoder(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        # Regressor to the classification labels
        self.fc_enconder = nn.Sequential(
            nn.Linear(256, num_output)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        
    def forward(self, x):
        # transform the input
        x = self.cnn_encoder(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 256)
        x = self.fc_enconder(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),  # b, 256, 7, 7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=(0,1)),  # b, 16, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=(0,1)),  # b, 16, 31, 31
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=(0,1)),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=(2,1), padding=1),  # b, 8, 63, 63
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=(2,1), padding=(0, 1)),  # b, 1, 128, 128
            nn.Tanh()
        )
        # Regressor to the classification labels
        self.fc_decoder = nn.Sequential(
            nn.Linear(num_inputs, 256)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        
    def forward(self, x):
        xs = self.fc_decoder(x)
        xs = xs.view(-1, 256, 1, 1)
        return self.cnn_decoder(xs)
        
class EmbeddingAEModel(nn.Module):
    def __init__(self, n_classes, snps_size, embedding_size):
        super().__init__()
        self.video_encoder = CNNEncoder(embedding_size)
        self.video_decoder = CNNDecoder(embedding_size)
        self.embedding_size = embedding_size
        self.snps_size = snps_size
        
        self.snp_mapper = nn.Sequential(
            nn.Linear(snps_size, 2048), 
            nn.Linear(2048, embedding_size)
        )
        self.classification = nn.Linear(embedding_size, n_classes)
        
        for m in self.snp_mapper.modules():
            weights_init_xavier(m)
            
        for m in self.classification.modules():
            weights_init_xavier(m)

    def forward(self, input_d):
        video_input, snps = input_d
        
        video_embedding = self.video_encoder(video_input)
        classification = self.classification(video_embedding)
        snps_embedding = self.snp_mapper(snps)
        
        video_decoded = self.video_decoder(video_embedding)
        return classification, video_embedding, snps_embedding, video_decoded
    
    
    def load_from_file(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()
        return self

class EmbeddingAEModelS(nn.Module):
    def __init__(self, n_classes, snps_size, embedding_size):
        super().__init__()
        self.video_encoder = CNNEncoder(embedding_size)
        self.video_decoder = CNNDecoder(embedding_size)
        self.embedding_size = embedding_size
        self.snps_size = snps_size
        
        self.snp_mapper = nn.Sequential(
            nn.Linear(snps_size, embedding_size)
        )
        self.classification = nn.Linear(embedding_size, n_classes)
        
        for m in self.snp_mapper.modules():
            weights_init_xavier(m)
            
        for m in self.classification.modules():
            weights_init_xavier(m)

    def forward(self, input_d):
        video_input, snps = input_d
        
        video_embedding = self.video_encoder(video_input)
        classification = self.classification(video_embedding)
        snps_embedding = self.snp_mapper(snps)
        
        video_decoded = self.video_decoder(video_embedding)
        return classification, video_embedding, snps_embedding, video_decoded
    
    
    def load_from_file(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()
        return self
