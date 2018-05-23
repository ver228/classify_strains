#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:55:57 2017

@author: ajaver
"""
from torch import nn
import torch.nn.functional as F

def weights_init_xavier(m):
    '''
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('Linear'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('BatchNorm2d'):
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

loss_funcs = dict(
        l2 = F.mse_loss,
        l1 = F.l1_loss
        )

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
#%%
class CNNClf(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.cnn_clf = nn.Sequential(
            nn.Conv2d(1, 32, 7),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1), 
            Flatten()
        )
        # Regressor to the classification labels
        self.fc_clf = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, num_output)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        
    def forward(self, x):
        # transform the input
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x


class CNNClf1D(nn.Module):
    def __init__(self, n_channels, num_output):
        super().__init__()
        self.cnn_clf = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(), 
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2), 
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1), 
            Flatten()
        )
        # Regressor to the classification labels
        self.fc_clf = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, num_output)
        )
        
        for m in self.modules():
            weights_init_xavier(m)
        
        
    def forward(self, x):
        if x.dim() == 4: #remove the first channel
            d = x.size()
            x = x.view(d[0], d[2], d[3])
        
        
        # transform the input
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x

#%%

class FullLoss(nn.Module):
    def __init__(self, embedding_loss_mixture=0.01, loss_type='l2'):
        super().__init__()

        self.embedding_loss_mixture = embedding_loss_mixture
        self.classification_loss = nn.CrossEntropyLoss()
        self.embedding_loss = loss_funcs[loss_type]
        
    def forward(self, embedding_output, target_classes):
        classification, video_embedding, snps_embedding = embedding_output
        classification_loss = self.classification_loss(classification,
                                                       target_classes)
        
        # Can't use the Loss layer here because it doesn't like - aej, likely due to autograd gradients
        _embedding_loss = self.embedding_loss(snps_embedding, video_embedding)
        
        loss = classification_loss + self.embedding_loss_mixture * _embedding_loss
        return loss

class EmbeddingModel(nn.Module):
    def __init__(self, video_model, n_classes, snps_size, embedding_size):
        super().__init__()
        self.video_model = video_model
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
        video_embedding = self.video_model(video_input)
        classification = self.classification(video_embedding)
        snps_embedding = self.snp_mapper(snps)
        return classification, video_embedding, snps_embedding

class EmbeddingModelDum(EmbeddingModel):
    '''
    Remove the embedding layer.  Use it for testing. 
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.snp_mapper = None
        
    def forward(self, input_d):
        video_input, snps = input_d
        video_embedding = self.video_model(video_input)
        classification = self.classification(video_embedding)
        return classification, video_embedding, video_embedding



def simple_no_emb(gen, embedding_size):
    video_model = CNNClf(embedding_size)
    
    
    model = EmbeddingModelDum(video_model, 
                   gen.n_classes, 
                   1, 
                   embedding_size)
    return model

def simple_w_emb(gen, embedding_size):
    video_model = CNNClf(embedding_size)
    
    
    model = EmbeddingModel(video_model, 
                   gen.n_classes, 
                   gen.n_snps, 
                   embedding_size)
    return model