#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""

import sys
import os
import torch
import time
from torch import nn

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, 'src')
sys.path.append(src_dir)

from flow import ROIFlowBatch
from models import VAE, AE, VAELoss
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

def main(model_name='AE', 
         n_epochs=1000,
         batch_size=32, 
         roi_size=128
         ):
    if model_name == 'VAE':
        model = VAE()
        criterion = VAELoss()
    elif 'AE':
        model = AE()
        criterion = nn.MSELoss()
    
    gen_details = ''
    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, gen_details, time.strftime('%Y%m%d_%H%M%S')))
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    if is_cuda:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    
    generator = ROIFlowBatch(mask_file, 
                             feat_file, 
                             is_cuda = is_cuda,
                             batch_size = batch_size, 
                             roi_size = roi_size)
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
    
    
    