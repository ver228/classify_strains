#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""

from flow import collate_fn, SkelTrainer
from path import get_path

import tqdm

from torch import nn
from torch.utils.data import DataLoader
from models import weights_init_xavier, Flatten


#https://arxiv.org/pdf/1705.09914.pdf
#open.ai video
#%%
def conv_layer(ni, nf, ks=3, stride=1):
    return nn.Sequential(
           nn.Conv2d(ni, nf, ks, bias = False, stride = stride, padding = ks//2),
           nn.BatchNorm2d(nf),
           nn.LeakyReLU(negative_slope = 0.1, inplace = True)
           )

class ResLayer(nn.Module):
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
                ] + [ResLayer(ch_in*2) for i in range(num_blocks)]
    
    
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
        
#%%
if __name__ == '__main__':
    import torch
    from models import CNNClf1D, CNNClf
    #set_type = 'skels'
    emb_set = 'AE_emb_20180206'
    
    fname = '/Users/avelinojaver/Documents/Data/experiments/classify_strains/CeNDR_{}.hdf5'.format(emb_set)
    gen = SkelTrainer(fname = fname,
                      is_divergent_set=True)
    
    batch_size = 8 
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        collate_fn = collate_fn,
                        num_workers = batch_size)
    gen.test()
    for ii, D in enumerate(tqdm.tqdm(loader)):
        X, target = D
        break
        
    
    #model = CNNClf1D(gen.embedding_size, gen.num_classes)    
    #pred =  model(X)
    #%%
    model = CNNClf(gen.num_classes)
    
    
    model_path = '/Volumes/rescomp1/data/WormData/experiments/classify_strains/results/log_divergent_set/angles_20180522_165626_simple_div_lr0.0001_batch8/model_best.pth.tar'
    #model_path = '/Volumes/rescomp1/data/WormData/experiments/classify_strains/results/log_divergent_set/angles_20180522_165626_simple_div_lr0.0001_batch8/checkpoint.pth.tar'
    
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    

    pred =  model(X)
    
    print(pred.max(1)[1], target)
    
    #
    #%%
    #model = Darknet([1, 3, 4, 6, 3], 254)