#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:00:00 2018

@author: ajaver
"""

from flow import FlowSampled, read_dataset_info
from models import AE2D
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

import os
import torch
import pickle
#%%
if __name__ == '__main__':
    is_cuda = False
    is_tiny = True
    size_per_epoch = 10#0
    batch_size = 2#2
    snippet_size = 255#255
    embedding_size = 32
    gpu_id = 0
    model = AE2D(embedding_size)
    
    #model_file = os.path.join(os.environ['HOME'], 'Github/classify_strains/results/AE2D__mix0.1L32_20180202_170055', 'checkpoint.pth.tar')
    model_file = os.path.join(os.environ['HOME'], 'Github/classify_strains/results/AE2D__mix0.1L32_20180202_170055', 'checkpoint.pth.tar')
    #model_file = os.path.join(os.environ['HOME'], 'Github/classify_strains/results/AE2D__mix0.0_trainL32_20180205_183813', 'checkpoint.pth.tar')
    #model_file = os.path.join(os.environ['HOME'], 'Github/classify_strains/results/AE2D__snippet5_trainL32_20180206_123614', 'checkpoint.pth.tar')
    
    
    
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    if is_cuda:
        model = model.cuda(gpu_id)
    #%%
    _set_divisions_file = '/data/ajaver/CeNDR_ROIs/set_divisions.p'
    with open( _set_divisions_file, "rb" ) as fid:
        set_divisions = pickle.load(fid)
    
    set_type = 'test'
    all_data = set_divisions[set_type]
    
    #all_data = read_dataset_info(is_tiny = is_tiny)
    generator = FlowSampled(all_data, 
                            size_per_epoch, 
                            batch_size, 
                            snippet_size, 
                            is_cuda=is_cuda, 
                            gpu_id=gpu_id)
    #%%
    nn = 0
    for S in generator:
        
        output, _ = model(S)
        #%%
        ss_embs = model.encoder(S).cpu()
        ss_embs = ss_embs.data.numpy()
        
        for ss in ss_embs:
            ss = ss.T
            plt.figure()
            plt.imshow(ss)
            plt.figure()
            plt.plot(ss[0])
            plt.plot(ss[15])
            plt.plot(ss[-1])
        #%%
        S = S.cpu()
        output = output.cpu()
        
        
        
        #%%
        dpi = 100
        def ani_frame(n_batch, save_name):
            #%%
            ori = S.data.squeeze().numpy()[n_batch]
            dat = output.data.squeeze().numpy()[n_batch]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            dd = np.hstack((ori[0], dat[0]))
            im = ax.imshow(dd,cmap='gray',interpolation='nearest')
            im.set_clim([0,1])
            fig.set_size_inches([8,4])
            plt.tight_layout()
            #%%
            def update_img(n):
                tmp = np.hstack((ori[n], dat[n]))
                im.set_data(tmp)
                return im
        
            #legend(loc=0)
            ani = animation.FuncAnimation(fig,update_img, 128, interval = 40)
            writer = animation.writers['ffmpeg'](fps = 25)
        
            ani.save(save_name,writer=writer,dpi=dpi)
            
            return ani
        
        for kk in range(S.size(0)):
            nn += 1
            save_name = '{}_demo{}.mp4'.format(set_type, nn)
            ani = ani_frame(kk, save_name)