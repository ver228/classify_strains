#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:43:03 2017

@author: ajaver
"""

import os
import sys
import time
import torch

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, os.pardir, 'src')
sys.path.append(src_dir)



from classify.trainer import init_generator, Trainer
import models
from models import FullLoss

#flag to check if cuda is available
is_cuda = torch.cuda.is_available()

def _get_log_dir(model_name, details=''):
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/recognize_worms/results'
    else:
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
    
    _BASEDIR = os.path.dirname(os.path.abspath(__file__))
    pdir = os.path.split(_BASEDIR)[-1]
    log_dir_root = os.path.join(log_dir_root, pdir)

    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, details, time.strftime('%Y%m%d_%H%M%S')))
    return log_dir


data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/_old/CeNDR_skel_smoothed.hdf5'
data_dir = None
if __name__ == '__main__':
    model_name = 'simple_no_emb'#'resnet18_no_emb'#
    data_file = data_dir
    is_reduced = True
    embedding_size = 256
    sample_size_seconds = 10
    sample_frequency_s = 0.04
    n_batch = 32
    n_epochs = 200
    embedding_loss_mixture = 0.001
    
    params = dict(
            is_reduced = is_reduced,
            dataset = 'CeNDR',
            data_file = data_file,
            sample_size_seconds = sample_size_seconds,
            sample_frequency_s = sample_frequency_s,
            n_batch = n_batch,
            transform_type = 'angles',
            is_normalized = False,
            is_cuda = is_cuda,
            is_return_snps = True,
            _valid_strains = ['N2', 'CB4856'] #used for testing
    )
    gen_details, train_generator, test_generator = init_generator(**params)
    
    assert model_name in dir(models)
    get_model_func = getattr(models, model_name)
    model = get_model_func(train_generator, embedding_size)
    
    log_dir = _get_log_dir(model_name, details=gen_details)
    #%%
    for X,Y in train_generator: break
    
    #%%
    #show some data for debugging purposes
    print(model)
    print(test_generator.valid_strains)
    print(log_dir)
    
    #maybe i should include the criterion and optimizer as input parameters
    criterion = FullLoss(embedding_loss_mixture)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if is_cuda:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    

    t = Trainer(model,
             optimizer,
             criterion,
             train_generator,
             test_generator,
             n_epochs,
             log_dir
             )
    t.fit()
    

  