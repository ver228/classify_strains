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
import numpy as np
import pickle


from flow import FlowSampled, read_dataset_info
from models import AE3D, AE2D, EmbRegLoss



#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, 'worm_autoencoder', 'src')
sys.path.append(src_dir)

from trainer import TrainerAutoEncoder


is_cuda = torch.cuda.is_available()
log_dir_root = os.path.join(os.environ['HOME'], 'Github/classify_strains/results')

_set_divisions_file = '/data/ajaver/CeNDR_ROIs/set_divisions.p'
with open( _set_divisions_file, "rb" ) as fid:
    set_divisions = pickle.load(fid)


def main(model_name='AE2D', 
         batch_size = 2,
         snippet_size = 255,
         roi_size = 128,
         n_epochs = 20000,
         size_per_epoch = 1000,
         embedding_size = 32,
         set_type = 'train',
         pretrained_path = '',#/home/ajaver@cscdom.csc.mrc.ac.uk/Github/classify_strains/results/AE2D__mix0.1L32_20180202_170055/checkpoint.pth.tar',
         emb_reg_loss_mix = 0.
         ):
    #%%
    assert set_type in ['', 'train', 'tiny', 'test']
    
    postfix_ = ''
    
    if model_name == 'AE3D':
        model = AE3D(embedding_size)
        criterion = nn.MSELoss()
    elif model_name == 'AE2D':
        model = AE2D(embedding_size)
        criterion = EmbRegLoss(emb_reg_loss_mix = emb_reg_loss_mix)
        postfix_ += '_mix{}'.format(emb_reg_loss_mix)
        
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print("Loading pretrained weigths", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        
    postfix_ = '_snippet{}'.format(snippet_size)
    postfix_ += '_' + set_type
    postfix_ += 'L{}'.format(embedding_size)
    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, postfix_, time.strftime('%Y%m%d_%H%M%S')))
    
    
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    if is_cuda:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    
    if set_type in set_divisions:
        all_data = set_divisions[set_type]
    else:
        all_data = read_dataset_info()
    
    
    generator = FlowSampled(all_data, size_per_epoch, batch_size, snippet_size, is_cuda=is_cuda)

    
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
    
