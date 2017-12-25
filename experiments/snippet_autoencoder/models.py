#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:08:01 2017

@author: ajaver
"""
import math
import torch
from torch import nn
from torch.autograd import Variable
    
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
            #nn.Sigmoid()
            nn.Tanh()
        )
        
        for m in self.modules():
            weights_init_xavier(m)
    
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

class AE2D(nn.Module):
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
            weights_init_xavier(m)
    
    def encoder(self, x):
        n_frames = x.size(2)
        roi_size = x.size(-1)
        x = x.view(-1, 1, roi_size, roi_size)
        x = self.cnn_encoder(x).view(-1, 256)
        x = self.fc_encoder(x)
        x = x.view(-1, n_frames, self.embedding_size)
        
        return x
        
    def decoder(self, x):
        n_frames = x.size(1)
        
        x = x.view(-1, self.embedding_size)
        x = self.fc_decoder(x)
        x = x.view(-1, 256, 1, 1)
        x = self.cnn_decoder(x)
        
        roi_size = x.size(-1)
        x = x.view(-1, 1, n_frames, roi_size, roi_size)
        return x
        
    def forward(self, x):
        embeddings = self.encoder(x)
        x_out = self.decoder(embeddings)
        return x_out, embeddings


class AE2D_RNN(AE2D):
    def __init__(self, 
                 embedding_size = 256, 
                 hidden_size = None, 
                 n_layer = 2):
        super().__init__(embedding_size=embedding_size)
        
        if hidden_size is None:
            hidden_size = max(64, embedding_size)
        
        self.embedding_size = embedding_size
        self.n_layer = n_layer
        
        self.encoder_lstm = nn.GRU(embedding_size, hidden_size, n_layer, batch_first=True)
        self.decoder_lstm = nn.GRU(embedding_size, hidden_size, n_layer, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_size, embedding_size)
        
        for m in self.modules():
            weights_init_xavier(m)
        
    def forward(self, x):
        cnn_embeddings = self.encoder(x)
        output_enc, hidden_enc = self.encoder_lstm(cnn_embeddings)
        
        if isinstance(hidden_enc, tuple):
            hidden_dec = tuple(x.clone() for x in hidden_enc)
        else:
            hidden_dec = hidden_enc.clone()
        
        #first input made it zeros
        n_batch, n_time, hidden_size = output_enc.size()
        input_dec_ = torch.zeros(n_batch, 1, self.embedding_size)
        if output_enc.is_cuda:
            input_dec_ = input_dec_.cuda()
        input_dec_ = Variable(input_dec_)
        #%%
        outputs_d = []
        for ii in range(n_time):
            output_dec, hidden_dec = self.decoder_lstm(input_dec_, hidden_dec)
            output_dec = self.decoder_linear(output_dec)
            input_dec_ = output_dec
            outputs_d.append(output_dec)
        
        #reverse data, it is easier to try to decode from the last layer to the first
        outputs_d = torch.cat(outputs_d[::-1], dim=1)
        
        x_out = self.decoder(outputs_d)
        return x_out, cnn_embeddings, hidden_enc




class EmbRegLoss(nn.Module):
    def __init__(self, emb_reg_loss_mix = 0.1):
        super().__init__()

        self.emb_reg_loss_mix = emb_reg_loss_mix
        self.mse_loss = nn.MSELoss()
        
        
    def emb_reg_loss(self, embeddings):
        n_frames = embeddings.size(1)
        dx = embeddings[:, 1:, :] - embeddings[:, :-1, :]
        _emb_reg_loss = dx.norm(p=2, dim=1).mean()/n_frames
        return _emb_reg_loss
    
    def forward(self, outputs, target):
        decoded_img = outputs[0]
        cnn_embeddings = outputs[1]
        
        d_loss = self.mse_loss(decoded_img, target)
        reg_loss = self.emb_reg_loss(cnn_embeddings)
        
        loss = d_loss + self.emb_reg_loss_mix * reg_loss
        return loss
    
if __name__ == '__main__':
    import os
    from flow import ROIFlowBatch
    
    data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/videos'
    fname = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    mask_file = os.path.join(data_dir,fname)
    feat_file = os.path.join(data_dir,fname.replace('.hdf5', '_featuresN.hdf5'))
    
    model = AE2D_RNN(embedding_size = 32)
    criterion = EmbRegLoss()
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             roi_size = 128,
                             batch_size = 2,
                             snippet_size = 255,
                             is_cuda = False
                             )
    
    #%%
    embedding_size = 32
    for S in generator:
        break
    print(S.size())
    
    outputs  = model(S)
    loss = criterion(outputs, S)
    
    #%%
    
#    x = model.fc_decoder(x)
#    x = x.view(-1, 256, 1, 1, 1)
#    print(x.size())
#    
#    xs = model.cnn_decoder(x)
#    print(xs.size())
    