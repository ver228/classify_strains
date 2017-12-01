#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:49:50 2017

@author: ajaver
"""
import os
import torch
import numpy as np
import matplotlib.pylab as plt
from train import AE, VAE, is_cuda, ROIFlowBatch
import glob


def results_AE(mask_file, feat_file):
    #%%
    model_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/worm_autoencoder'
    dnames = glob.glob(os.path.join(model_dir_root, 'AE_L*'))
    
    all_models = []
    for d in dnames:
        embedding_size = int(d.split('AE_L')[-1].partition('_')[0])
        model_path = os.path.join(d, 'checkpoint.pth.tar')
        print(embedding_size)
        model = AE(embedding_size)
        
        
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
         
        all_models.append((embedding_size, model))
        
    
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             is_cuda = is_cuda,
                             batch_size = 32, 
                             roi_size = 128,
                             is_shuffle = True
                             )
    
    for X in generator:
        break
    cx = X.data.squeeze().numpy()
    
    all_imgs = []
    for n, mod in all_models:
        decoded = mod.forward(X)
        imgs = decoded.data.squeeze().numpy()
        all_imgs.append((n, imgs))
    
    all_imgs = sorted(all_imgs, key = lambda x : x[0])
#    #%%
#    m, mod = all_models[0]
#    embedding =  mod.encoder(X)
#    decoded = mod.decoder(embedding)
#    
#    n1,n2 = 1, 25
#    xd = decoded.data.squeeze().numpy()
#    
#    plt.figure(figsize=(15, 5))
#    plt.subplot(1,4,1)
#    plt.imshow(xd[n1], interpolation=None, cmap='gray')
#    plt.title('A')
#    
#    plt.subplot(1,4,2)
#    plt.imshow(xd[n2], interpolation=None, cmap='gray')
#    plt.title('B')
#    
#    plt.subplot(1,4,3)
#    z = (embedding[n1]+embedding[n2]).view(1, -1)
#    zd = mod.decoder(z).data.squeeze().numpy()
#    plt.imshow(zd, interpolation=None, cmap='gray')
#    plt.title('A + B')
#    
#    plt.subplot(1,4,4)
#    z = (embedding[n1]-embedding[n2]).view(1, -1)
#    zd = mod.decoder(z).data.squeeze().numpy()
#    plt.imshow(zd, interpolation=None, cmap='gray')
#    plt.title('A - B')
    
    
    
    #%%
    n_models = len(all_imgs)
    for ii, ori in enumerate(cx):
        plt.figure(figsize=(5*(n_models+1), 5))
        plt.subplot(1,n_models+1,1)
        plt.imshow(ori, interpolation=None, cmap='gray')
        for n_i, (n, imgs) in enumerate(all_imgs):
            plt.subplot(1,n_models+1, n_i+2)
            plt.imshow(imgs[ii], interpolation=None, cmap='gray')
            plt.title(n)
        plt.savefig('AE_{}.png'.format(ii))
        
def results_VAE(mask_file, feat_file):
    #model_path = os.path.join(model_dir_root, 'VAE_L64_20171124_144333', 'checkpoint.pth.tar')
    #model = VAE(64)
    #%%
    model_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/worm_autoencoder'
    dnames = glob.glob(os.path.join(model_dir_root, 'VAE_L*'))
    
    all_models = []
    for d in dnames:
        embedding_size = int(d.split('VAE_L')[-1].partition('_')[0])
        model_path = os.path.join(d, 'checkpoint.pth.tar')
        print(embedding_size)
        model = VAE(embedding_size)
        
        
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
         
        all_models.append((embedding_size, model))
        
    
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             is_cuda = is_cuda,
                             batch_size = 32, 
                             roi_size = 128,
                             is_shuffle = True
                             )
    
    for X in generator:
        break
    cx = X.data.squeeze().numpy()
    
    all_imgs = []
    for n, mod in all_models:
        decoded = mod.forward(X)[0]
        imgs = decoded.data.squeeze().numpy()
        all_imgs.append((n, imgs))
    
    all_imgs = sorted(all_imgs, key = lambda x : x[0])
    #%%
    m, mod = all_models[0]
    embedding =  mod.encoder(X)[0]
    decoded = mod.decoder(embedding)
    
    n1,n2 = 5, 3
    xd = decoded.data.squeeze().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1,4,1)
    plt.imshow(xd[n1], interpolation=None, cmap='gray')
    plt.title('A')
    
    plt.subplot(1,4,2)
    plt.imshow(xd[n2], interpolation=None, cmap='gray')
    plt.title('B')
    
    plt.subplot(1,4,3)
    z = (embedding[n1]+embedding[n2]).view(1, -1)
    zd = mod.decoder(z).data.squeeze().numpy()
    plt.imshow(zd, interpolation=None, cmap='gray')
    plt.title('A + B')
    
    plt.subplot(1,4,4)
    z = (embedding[n1]-embedding[n2]).view(1, -1)
    zd = mod.decoder(z).data.squeeze().numpy()
    plt.imshow(zd, interpolation=None, cmap='gray')
    plt.title('A - B')
    #%%
    n_models = len(all_imgs)
    for ii, ori in enumerate(cx):
        plt.figure(figsize=(5*(n_models+1), 5))
        plt.subplot(1,n_models+1,1)
        plt.imshow(ori, interpolation=None, cmap='gray')
        for n_i, (n, imgs) in enumerate(all_imgs):
            plt.subplot(1,n_models+1, n_i+2)
            plt.imshow(imgs[ii], interpolation=None, cmap='gray')
            plt.title(n)        
        plt.savefig('VAE_{}.png'.format(ii))
        
    
        #%%
if __name__ == '__main__':
    from train import mask_file
    #mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_190517/QX1794_worms10_food1-10_Set5_Pos4_Ch4_19052017_125059.hdf5'
    #mask_file = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/control_pulse/pkd2_2min_Ch1_11052017_120432.hdf5'
    feat_file = mask_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_featuresN.hdf5')


    results_VAE(mask_file, feat_file)
    
    #decoded, mu, _ = model.forward(X)
    #z, _, _ = model.encoder(X)
    #decoded = model.decoder(z*2)
    
    