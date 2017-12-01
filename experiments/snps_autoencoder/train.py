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
from torch import nn
import random

import tqdm

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.trainer import init_generator, Trainer
from models import SNPMapperVAE, VAELoss
import numpy as np

if __name__ == '__main__':
    n_batch = 32
    embedding_size = 64
    
    data_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/_old/CeNDR_skel_smoothed.hdf5'
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
            data_file = data_file,
            is_reduced = False,
            dataset = 'CeNDR',
            n_batch = n_batch,
            is_cuda = is_cuda,
            is_return_snps = True,
            _valid_strains = None #used for testing
    )
    _, train_generator, test_generator = init_generator(**params)
    
    model = SNPMapperVAE(embedding_size, train_generator.n_snps)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    #%%
    n_epochs = 10000
    strains = train_generator.valid_strains
    pbar = tqdm.trange(n_epochs)
    for epoch in pbar:
        random.shuffle(strains)
        #%%
        X = train_generator.snps_data[strains].astype(np.float32).values
        X = torch.autograd.Variable(torch.from_numpy(X.T))
        Xd, mu, logvar = model(X)
        #%%
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse = criterion(Xd, X)
        loss = kld + mse
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #%%
        dd = '{}, mse {}, kld {}'.format(epoch, mse.data[0], kld.data[0])
        pbar.set_description(dd, refresh=False)
        


#    t = Trainer(model,
#             optimizer,
#             criterion,
#             train_generator,
#             test_generator,
#             n_epochs,
#             log_dir
#             )
#    t.fit()
#    
#if __name__ == '__main__':
#    import fire
#    fire.Fire(main)  
 