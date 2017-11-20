#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:38:53 2017

@author: ajaver
"""

import torch
from torch import nn
import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.models.resnet import ResNetS, Bottleneck
from classify.flow import SkeletonsFlowFull, get_valid_strains, get_datset_file

if __name__ == '__main__':
    import numpy as np
    
    models_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/model_20171103/'
    model_name = 'resnet50_R_CeNDR_ang__S10_F0.04_20171104_182812.pth.tar'
    dataset = 'CeNDR'
    n_classes = 197
    
    
    fname = os.path.join(models_path, model_name)
    model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes = n_classes)
    checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
#    
#    
#    data_file = get_datset_file(dataset)
#    valid_strains = get_valid_strains(dataset, is_reduced=True)
#    
#    dd = model_name.split('_')
#    sample_size_seconds = [float(x[1:]) for x in dd if x.startswith('S')][0]
#    sample_frequency_s = [float(x[1:]) for x in dd if x.startswith('F')][0]
    #%%
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            print(m.weight.data)
            break
        
    #%%
    first_layer = m.weight.data.numpy()
    first_layer = np.squeeze(first_layer)
    
    import matplotlib.pylab as plt
    plt.figure()
    for ii in range(first_layer.shape[0]):
        plt.subplot(8,8, ii+1)
        plt.imshow(first_layer[ii], interpolation='none')
    