#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:28:35 2017

@author: ajaver
"""
import torch
from torch import nn
import time
import os
import sys

from flow import ROIFlowBatch
from models import AE3D, AE2D, AE2D_RNN, EmbRegLoss

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, 'worm_autoencoder', 'src')
sys.path.append(src_dir)

from trainer import TrainerAutoEncoder


is_cuda = torch.cuda.is_available()
if sys.platform == 'linux':
    data_dir = os.environ['TMPDIR']
else:        
    data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/videos'


if sys.platform == 'linux':
    log_dir_root = '/work/ajaver/classify_strains/results'
else:        
    log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'

#flag to check if cuda is available
is_cuda = torch.cuda.is_available()

#add the parent directory to the log results
pdir = os.path.split(_BASEDIR)[-1]
log_dir_root = os.path.join(log_dir_root, pdir)

fname = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
mask_file = os.path.join(data_dir,fname)
feat_file = os.path.join(data_dir,fname.replace('.hdf5', '_featuresN.hdf5'))

def main(model_name='AE2D_RNN', 
         batch_size = 2,
         snippet_size = 255,
         roi_size = 128,
         n_epochs = 10000,
         embedding_size = 32,
         max_n_frames = -1,
         pretrained_path = None,
         emb_reg_loss_mix = 0.
         ):
    #%%
    if 'AE3D':
        model = AE3D(embedding_size)
        criterion = nn.MSELoss()
    elif 'AE2D':
        model = AE2D(embedding_size)
        criterion = EmbRegLoss(emb_reg_loss_mix=emb_reg_loss_mix)
    elif 'AE2D_RNN':
        model = AE2D_RNN(embedding_size, hidden_size = 256, n_layer = 2)
        criterion = EmbRegLoss(emb_reg_loss_mix=emb_reg_loss_mix)
    
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print("Loading pretrained weigths", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        
    
    
    if max_n_frames> 0:
        dd = '_tiny{}'.format(max_n_frames)
    else:
        dd = ''
    
    details = dd + 'L{}'.format(embedding_size)
    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, details, time.strftime('%Y%m%d_%H%M%S')))
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    if is_cuda:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             roi_size = roi_size,
                             batch_size = batch_size,
                             snippet_size = snippet_size,
                             is_cuda = is_cuda,
                             size_per_epoch = 1000,
                             max_n_frames = max_n_frames
                             )
    
    t = TrainerAutoEncoder(
                 model,
                 optimizer,
                 criterion,
                 generator,
                 n_epochs,
                 log_dir
                 )
    t.fit()

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    
