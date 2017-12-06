#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:05:46 2017

@author: ajaver
"""

import os
import sys
import torch
import matplotlib.pylab as plt

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.flow import SkeletonsFlowShuffled, SkeletonsFlowFull, get_valid_strains, get_datset_file
from models import EmbeddingAEModel

def load_model(model_path, n_classes, n_snps, embedding_size):
    model = EmbeddingAEModel(n_classes, n_snps, embedding_size)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/'
    model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L32_Clf1_Emb0.1_AE1_20171129_001223_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L256_Clf1_Emb0.1_AE1_20171129_001337_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L32_Clf0_Emb0.1_AE1_20171129_001219_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L256_Clf0_Emb0.1_AE1_20171129_001213_checkpoint.pth.tar')
    
    
    bn = os.path.basename(model_path)
    parts = bn.split('_')    
    embedding_size = [int(x[1:]) for x in parts if x.startswith('L')][0]
    
    dataset = 'CeNDR'
    valid_strains = get_valid_strains(dataset, is_reduced=True)
    data_file = get_datset_file(dataset)
    
    #data_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/_old/CeNDR_skel_smoothed.hdf5'
    gen = SkeletonsFlowShuffled(
            set_type = 'train',
            data_file = data_file,
            sample_size_seconds = 10,
            sample_frequency_s = 0.04,
            n_batch = 4,
            is_return_snps = True,
            is_autoencoder = True,
            valid_strains = valid_strains
    )
    
    model = load_model(model_path, gen.n_classes, gen.n_snps, embedding_size)
    #%%
    for X,Y in gen:
        output = model(X)
        break
    #%%
    in_v = X[0].data.squeeze().numpy()
    out_v = output[-1].data.squeeze().numpy()
    #%%
    for nn in range(in_v.shape[0]):
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(in_v[nn].T, interpolation='none')
        plt.subplot(2,1,2)
        plt.imshow(out_v[nn].T, interpolation='none')
    
    
    
    