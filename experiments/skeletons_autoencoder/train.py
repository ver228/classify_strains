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
from models import EmbeddingAEModel, FullAELoss


#data_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/_old/CeNDR_skel_smoothed.hdf5'
def main(
    data_file = None, #get defaults
    is_reduced = True,
    embedding_size = 256,
    sample_size_seconds = 10,
    sample_frequency_s = 0.04,
    n_batch = 32,
    n_epochs = 200,
    embedding_loss_mixture = 0.1,
    classification_loss_mixture = 1.,
    autoencoder_loss_mixture = 1.
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
            is_autoencoder = True,
            _valid_strains = None #used for testing
    )
    _, train_generator, test_generator = init_generator(**params)
    
    model = EmbeddingAEModel(train_generator.n_classes, 
                           train_generator.n_snps, 
                           embedding_size)
    
    criterion = FullAELoss(
                classification_loss_mixture = classification_loss_mixture,
                embedding_loss_mixture = embedding_loss_mixture, 
                autoencoder_loss_mixture = autoencoder_loss_mixture
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_mixes = (classification_loss_mixture, embedding_loss_mixture, autoencoder_loss_mixture)
    
    extra_details = 'L{}_M{}_{}_{}'.format(embedding_size, *loss_mixes)
    if is_reduced:
        extra_details = 'R_' + extra_details
    log_dir = os.path.join(log_dir_root, '{}_{}_{}'.format(model_name, extra_details, time.strftime('%Y%m%d_%H%M%S')))
    
    #show some data for debugging purposes
    print(model)
    print(test_generator.valid_strains)
    print(log_dir)
    
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
 