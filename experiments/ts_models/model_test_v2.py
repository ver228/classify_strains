#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import os
import torch
from path import get_path

from models import CNNClf1D, CNNClf

from flow import collate_fn, SkelTrainer
import tqdm
from torch.utils.data import DataLoader

from train import get_predictions
#%%
if __name__ == '__main__':
    #set_type = 'skels'
    set_type = 'AE_emb_20180206'
    
    fname, results_dir_root = get_path(set_type)
    
    model_path_str = os.path.join(results_dir_root, 'log_divergent_set/angles_20180523_132033_simple_div_lr0.0001_batch8/checkpoint_{}.pth.tar')
    
    
    states = []
    for epoch in range(0, 100, 5):
        model_path = model_path_str.format(epoch)
        state = torch.load(model_path, map_location = 'cpu')
        print(state['epoch'])
        W = state['state_dict']['cnn_clf.0.weight'][0].squeeze().numpy()
        plt.figure()
        plt.imshow(W)
        
        states.append(state)
     