#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:31:12 2018

@author: avelinojaver
"""
import cv2
    
import os
import torch
from path import get_path

from models import SimpleDilated

from flow import collate_fn, SkelTrainer
import tqdm
from torch.utils.data import DataLoader

from train import get_predictions
#%%
def apply_mask(image_gray, mask, color, alpha=0.5):
    image = np.repeat(image_gray[..., None], 3, axis=2)
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] * (1 - alpha) + 
                                  alpha * color[c] * mask ,
                                  image[:, :, c])
    return image

#%%
if __name__ == '__main__':
    set_type = 'angles'
    #set_type = 'AE_emb_20180206'
    
    fname, results_dir_root = get_path(set_type)
    
    #model_path = os.path.join(results_dir_root, 'log_divergent_set/angles_20180522_165626_simple_div_lr0.0001_batch8/model_best.pth.tar')
    model_path = os.path.join(results_dir_root, 'log_divergent_set/angles_20180524_115242_simpledilated_div_lr0.0001_batch8/model_best.pth.tar')
    
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
    
    
    model = SimpleDilated(gen.num_classes)
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
         
        
        
        M = model.cnn_clf(X)
        pred =  model.fc_clf(M)
        _, pred_l = pred.max(1)
        
        all_res.append(get_predictions(pred, target))
        
        ytrue, ypred = all_res[-1][:2]
        print('T: ', ytrue)
        print('P: ', ypred)
        
        break
    #%%
    
    for ii in range(batch_size):
        ind = target[ii].item()
        
        maps = M.detach().cpu().numpy()
        
        
        FC = model.fc_clf[2]
        W = FC.weight[ind, :]
        B = FC.bias[ind]
        
        maps_r = (M*W.view(1, -1, 1, 1) + B).sum(1).detach().numpy()
        bot = maps_r.min()
        top = maps_r.max()
        
        maps_r = (maps_r-bot)/(top - bot)
        
        
        plt.figure()
        for nn, (tt, pp) in enumerate(zip(target, pred_l)):
            mm = maps_r[nn]
            
            
            strC = 'r' if tt == ind else 'gray'
            
            plt.plot(np.sum(mm, axis=1), strC)
        
        strT = 'T{} P{}'.format(target[ii], pred_l[ii])
        plt.title(strT)
    #%%
    for ii in range(batch_size):
        ind = target[ii].item()
        maps = M.detach().cpu().numpy()
        
        
        FC = model.fc_clf[2]
        W = FC.weight[ind, :]
        B = FC.bias[ind]
        
        maps_r = (M*W.view(1, -1, 1, 1) + B).sum(1).detach().numpy()
        bot = maps_r.min()
        top = maps_r.max()
        
        maps_r = (maps_r-bot)/(top - bot)
        
        
    
        fig, axs = plt.subplots(batch_size, 1, sharex=True, sharey=True)
        for nn, (tt, pp) in enumerate(zip(target, pred_l)):
            img = X[nn].squeeze().numpy()/(2*np.pi) + 0.5
            
            mm = maps_r[nn]
            
            mm_r = cv2.resize(mm, img.shape[::-1]) 
            img_rgb = apply_mask(img, mm_r, (1, 0, 0), alpha=0.5)
            
            #plt.figure(figsize=(20, 8))
            axs[nn].imshow(img_rgb, aspect = 'auto')
            
            
            #strT = '{} : T{} P{}'.format(nn, tt, pp)
            #plt.title(strT)
        
    