#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:00:00 2018

@author: ajaver
"""

from flow import FlowSampled
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
    size_per_epoch = 10
    batch_size = 2
    snippet_size = 255#255
    
    gpu_id = 0
    
    save_dir = './results'
    model_path_root = os.path.join(os.environ['HOME'], 'Github/classify_strains/results/')
    #model_names = None
    model_names = ['AE2D__snippet5_trainL64_20180208_112321', 
                   'AE2D__snippet5_trainL32_20180206_123614',
                   'AE2D__snippet5_trainL32_20180208_152640'
                   ]
        
    
    #test all the models in the model_path_root
    if model_names is None:
        model_names = os.listdir(model_path_root)
        
    
    #load all models
    all_models = []
    for dname in model_names:
        model_file = os.path.join(model_path_root, dname, 'checkpoint.pth.tar')
        if os.path.exists(model_file):
            embedding_size = int(dname.split('trainL')[-1].split('_')[0])
            model =  AE2D(embedding_size)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            if is_cuda:
                model = model.cuda(gpu_id)
    
            all_models.append((dname, model))
            
    #create the directory where the files are going to be saved
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    #%%
    _set_divisions_file = '/data/ajaver/CeNDR_ROIs/set_divisions.p'
    with open( _set_divisions_file, "rb" ) as fid:
        set_divisions = pickle.load(fid)
    
    set_type = 'test'
    all_data = set_divisions[set_type]
    
    generator = FlowSampled(all_data, 
                            size_per_epoch, 
                            batch_size, 
                            snippet_size, 
                            is_cuda=is_cuda, 
                            gpu_id=gpu_id)
    #%%
    for ii_s, S in enumerate(generator):
        for model_name, model in all_models:
            output, _ = model(S)
            #%%
            ss_embs = model.encoder(S).cpu()
            ss_embs = ss_embs.data.numpy()
            
            for ss in ss_embs:
                ss = ss.T
                #%%
                plt.figure(figsize=(8,8))
                plt.subplot(2,1,1)
                plt.imshow(ss)
                plt.subplot(2,1,2)
                plt.plot(ss[0])
                plt.plot(ss[15])
                plt.plot(ss[-1])
                #$%%
            #%%
            S_c = S.cpu()
            output_c = output.cpu()
            
            
            
            #%%
            dpi = 100
            def ani_frame(n_batch, save_name):
                #%%
                ori = S_c.data.squeeze().numpy()[n_batch]
                dat = output_c.data.squeeze().numpy()[n_batch]
                
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
            
            
            
            for kk in range(S_c.size(0)):
                nn = ii_s*S_c.size(0) + kk
                ff = '{}_demo{}_{}.mp4'.format(set_type, nn, model_name)
                save_name = os.path.join(save_dir, ff) 
                ani = ani_frame(kk, save_name)