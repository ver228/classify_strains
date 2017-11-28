#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:28:35 2017

@author: ajaver
"""
import torch
from flow import ROIFlowBatch
from torch import nn
from models import AE3D
import time
import os
import sys
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

def main(model_name='AE3D', 
         batch_size = 4,
         snippet_size = 128,
         roi_size = 128,
         n_epochs = 1000,
         embedding_size = 256
         ):
    #%%
    if 'AE3D':
        model = AE3D(embedding_size)
    criterion = nn.MSELoss()
    
    details = 'L{}'.format(embedding_size)
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
                             is_cuda = is_cuda
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
    
