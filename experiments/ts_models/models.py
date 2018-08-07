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

#%%

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
#https://arxiv.org/pdf/1705.09914.pdf
#open.ai video

def conv_layer(ni, nf, ks=3, stride=1, dilation=1):
    if isinstance(ks, (float, int)):
        ks = (ks, ks)
    
    if isinstance(dilation, (float, int)):
        dilation = (dilation, dilation)
    
    pad = [x[0]//2*x[1] for x in zip(ks, dilation)]
    
    return nn.Sequential(
            
            
           nn.Conv2d(ni, nf, ks, bias = False, stride = stride, padding = pad, dilation = dilation),
           nn.BatchNorm2d(nf),
           nn.LeakyReLU(negative_slope = 0.1, inplace = True)
           )

class ResLayerBottleNeck(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer(ni, ni*2, ks = 1) #bottleneck
        self.conv2 = conv_layer(ni*2, ni, ks = 3)
    
    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out

    
class Darknet(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride = 1):
        return [conv_layer(ch_in, ch_in*2, stride = stride)
                ] + [ResLayerBottleNeck(ch_in*2) for i in range(num_blocks)]
    
    
    def __init__(self, num_blocks, num_classes, nf = 32):
        super().__init__()
        layers = [conv_layer(1, nf, ks=3, stride=1)]
        
        for i, nb in enumerate(num_blocks):
            #this is not part of dark net, but I want to reduce the size of the model
            #otherwise I start to have problems with memory
            layers += [nn.MaxPool2d((1, 4))]
            layers += self.make_group_layer(nf, nb, stride = 2)
            nf *= 2
        
        
        layers += [nn.AdaptiveAvgPool2d(1),  Flatten()]
        self.cnn_clf = nn.Sequential(*layers)
        
        self.fc_clf = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(nf, 32),
                
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(32, num_classes)
                )
        
        for m in self.modules():
            weights_init_xavier(m)
        
    def forward(self, x):
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x
#%% ResNet dilated

class ResLayer(nn.Module):
    def __init__(self, ni, ks = 3, dilation=1):
        super().__init__()
        self.conv1 = conv_layer(ni, ni, ks = ks, dilation = dilation) 
        self.conv2 = conv_layer(ni, ni, ks = ks, dilation = dilation)
    
    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out  

class DilatedResNet(nn.Module):
    #https://github.com/fyu/drn/blob/master/drn.py
    def make_group_layer(self, ch_in, num_blocks, stride = 1, dilation=1):
        return [conv_layer(ch_in, ch_in*2, stride = stride, dilation = dilation)
                ] + [ResLayer(ch_in*2, dilation = dilation) for i in range(num_blocks)]
    
    
    def __init__(self, num_blocks, num_classes, nf = 16, dropout_fc = 0.5):
        super().__init__()
        layers = [conv_layer(1, nf, ks = 7, stride=1)]
        
        
        dilation = 1
        for ii, nb in enumerate(num_blocks):
            if ii < 2:
                stride = (2,2)
            else:
                stride = (1,2)
            
            if ii >= len(num_blocks) - 2:
                dilation *= 2
            
            #this is not part of dark net, but I want to reduce the size of the model
            #otherwise I start to have problems with memory
            layers += self.make_group_layer(nf, nb, stride = stride, dilation = (1, dilation))
            nf *= 2
        
        #nf = nf/2
            
        layers += [conv_layer(nf, nf, ks = 3, dilation = (1, 2)),
        conv_layer(nf, nf, ks = 3, dilation = 2)]
        
        layers += [conv_layer(nf, nf, ks = 3, dilation = 1),
        conv_layer(nf, nf, ks = 3, dilation = 1)]
        
        
        
        
        self.cnn_clf = nn.Sequential(*layers)
        
        self.fc_clf = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Dropout(dropout_fc),
                nn.Conv2d(nf, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True),
                Flatten()
                )
        
        for m in self.modules():
            weights_init_xavier(m)
    
    def forward(self, x):
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x

def drn111111(num_classes):
    return DilatedResNet([1, 1, 1, 1, 1, 1], num_classes)    

#%%
class SimpleDilated(nn.Module):
    #https://github.com/fyu/drn/blob/master/drn.py
    
    def __init__(self, num_classes, nf = 16, dropout_fc = 0.5):
        super().__init__()
        layers = [conv_layer(1, nf, ks = 7, stride=1)]
        
        
        num_blocks = 5
        for ii in range(num_blocks):
            if ii < 2:
                stride = (2,2)
            else:
                stride = (1,2)
            
            layers += conv_layer(nf, nf*2, stride = stride, dilation = 1)
            nf *= 2
        
            
        layers += [
                conv_layer(nf, nf, ks = 3, dilation = (1, 2)),
                conv_layer(nf, nf, ks = 3, dilation = (1, 4)),
                conv_layer(nf, nf, ks = 3, dilation = (1, 2)),
                conv_layer(nf, nf, ks = 3, dilation = (1, 1))
                ]
        
        
        
        
        self.cnn_clf = nn.Sequential(*layers)
        
        self.fc_clf = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Dropout(dropout_fc),
                nn.Conv2d(nf, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True),
                Flatten()
                )
        
        for m in self.modules():
            weights_init_xavier(m)
    
    def forward(self, x):
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x    
#%%
def conv_layer1d(ni, nf, ks=3, stride=1, dilation=1):
    pad = ks//2*dilation
    return nn.Sequential(
            
            
           nn.Conv1d(ni, nf, ks, bias = False, stride = stride, padding = pad, dilation = dilation),
           nn.BatchNorm1d(nf),
           nn.LeakyReLU(negative_slope = 0.1, inplace = True)
           )
class SimpleDilated1D(nn.Module):
    #https://github.com/fyu/drn/blob/master/drn.py
    
    def __init__(self, embedding_size, num_classes, nf = 32, dropout_fc = 0.5):
        super().__init__()
        layers = [conv_layer1d(embedding_size, nf, ks = 7, stride=1),
                  conv_layer1d(nf, nf, ks = 7, stride=1)
                  ]
        
        
        num_blocks = 5
        stride = 2
        for ii in range(num_blocks):
            layers += conv_layer1d(nf, nf*2, stride = stride, dilation = 1)
            nf *= 2
        
            
        layers += [
                conv_layer1d(nf, nf, ks = 3, dilation = 2),
                conv_layer1d(nf, nf, ks = 3, dilation = 4),
                conv_layer1d(nf, nf, ks = 3, dilation = 2),
                conv_layer1d(nf, nf, ks = 3, dilation = 1)
                ]
        
        
        
        
        self.cnn_clf = nn.Sequential(*layers)
        
        self.fc_clf = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  
                nn.Dropout(dropout_fc),
                nn.Conv1d(nf, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True),
                Flatten()
                )
        
        for m in self.modules():
            weights_init_xavier(m)
    
    def forward(self, x):
        if x.dim() == 4: #remove the first channel
            d = x.size()
            x = x.view(d[0], d[2], d[3])
        x = self.cnn_clf(x)
        x = self.fc_clf(x)
        return x    
    
#%%            
if __name__ == '__main__':
    import os
    import torch
    from path import get_path
    
    from flow import collate_fn, SkelTrainer
    import tqdm
    from torch.utils.data import DataLoader
    
    
    
    #set_type = 'angles'
    set_type = 'AE_emb_20180206'
    
    fname, results_dir_root = get_path(set_type)
    
    cuda_id = 0
    if torch.cuda.is_available():
        dev_str = "cuda:" + str(cuda_id)
        print("THIS IS CUDA!!!!")
        
    else:
        dev_str = 'cpu'
      
    print(dev_str)
    device = torch.device(dev_str)
    
    gen = SkelTrainer(fname = fname,
                      is_divergent_set=True)
    
    #%%
    model = SimpleDilated1D(gen.embedding_size,gen.num_classes)
    model = model.to(device)
    
    #%%
    batch_size = 1
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        collate_fn = collate_fn,
                        num_workers = batch_size)
    #%%
    all_res = []
    pbar = tqdm.tqdm(loader)
    for x_in, y_in in pbar:
        X = x_in.to(device)
        target =  y_in.to(device)
        break 
    #%%
    pred = model(X)
        