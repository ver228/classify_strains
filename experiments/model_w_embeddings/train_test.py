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
from classify.models.model_w_embedding import FullLossL2
import models

def main(
    model_name = 'resnet18_w_embedding',
    data_file = None, #get defaults
    is_reduced = False,
    embedding_size = 256,
    sample_size_seconds = 10,
    sample_frequency_s = 0.04,
    n_batch = 32,
    n_epochs = 200,
    
    ):
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
    
    #flag to check if cuda is available
    is_cuda = torch.cuda.is_available()
    
    #add the parent directory to the log results
    pdir = os.path.split(_BASEDIR)[-1]
    log_dir_root = os.path.join(log_dir_root, pdir)
    
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
            _valid_strains = None #used for testing
    )
    gen_details, train_generator, test_generator = init_generator(**params)
    
    
    assert model_name in dir(models)
    get_model_func = getattr(models, model_name)
    model = get_model_func(train_generator, embedding_size, embedding_loss_mixture)
    
    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, gen_details, time.strftime('%Y%m%d_%H%M%S')))
    
    #show some data for debugging purposes
    print(model)
    print(test_generator.valid_strains)
    print(log_dir)
    
    #maybe i should include the criterion and optimizer as input parameters
    criterion = FullLossL2()
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
    
if __name__ == '__main__':
  import fire
  fire.Fire(main)  
  