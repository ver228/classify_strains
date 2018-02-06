#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 00:24:15 2017

@author: ajaver
"""
import matplotlib.pylab as plt
import matplotlib.animation as animation
import numpy as np
import os
import torch

from flow import ROIFlowBatch

from models import AE3D

if __name__ == '__main__':
    
    data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/videos'
    fname = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
    mask_file = os.path.join(data_dir,fname)
    feat_file = os.path.join(data_dir,fname.replace('.hdf5', '_featuresN.hdf5'))
    
    #model_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/snippet_autoencoder/AE3D__tiny2L256_20171129_171930/checkpoint.pth.tar'
    #model_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/snippet_autoencoder/AE3D_L256_20171128_150429/checkpoint.pth.tar'
    #model_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/snippet_autoencoder/AE3D_L256_20171129_155641/checkpoint.pth.tar'
    model_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/snippet_autoencoder/AE3D_L256_20171206_145027/checkpoint.pth.tar'
    model = AE3D(256)
    
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             roi_size = 128,
                             batch_size = 4,
                             snippet_size = 255,
                             max_n_frames=-1
                             )
    import time
    tic = time.time()
    for ii, S in enumerate(generator):
        print('G', time.time()-tic)
        tic = time.time()
        #if ii > 10: break
        
        output = model(S)
        print('M', time.time()-tic)
        tic = time.time()
        break
    
    
    #%%
    dpi = 100
    def ani_frame(n_batch):
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
    
        ani.save('demo{}.mp4'.format(n_batch),writer=writer,dpi=dpi)
        return ani
    
    for kk in range(S.size(0)):
        ani = ani_frame(kk)
        plt.show()