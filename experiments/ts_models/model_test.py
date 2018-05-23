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
    set_type = 'angles'
    #set_type = 'AE_emb_20180206'
    
    fname, results_dir_root = get_path(set_type)
    
    model_path = os.path.join(results_dir_root, 'log_divergent_set/angles_20180522_165626_simple_div_lr0.0001_batch8/model_best.pth.tar')
    
    #model_path = os.path.join(results_dir_root, 'log_divergent_set/angles_20180523_142824_simple_div_lr0.0001_batch8/checkpoint_4.pth.tar')
    
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
    
    
    model = CNNClf(gen.num_classes)
    model = model.to(device)
    
    assert set_type in model_path
    state = torch.load(model_path, map_location = dev_str)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #aw = np.load(model_path.replace('.pth.tar', '.npy'))
    
    #%%
    batch_size = 8 
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        collate_fn = collate_fn,
                        num_workers = batch_size)
    gen.test()
    #%%
    all_res = []
    pbar = tqdm.tqdm(loader)
    for x_in, y_in in pbar:
        X = x_in.to(device)
        target =  y_in.to(device)
         
        
        pred =  model(X)
        
        all_res.append(get_predictions(pred, target))
    
        ytrue, ypred = all_res[-1][:2]
        print('T: ', ytrue)
        print('P: ', ypred)
        
     