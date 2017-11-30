#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:02:55 2017

@author: ajaver
"""
import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
import os
import sys
import torch
import seaborn as sns

src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src')
sys.path.append(src_dir)
from classify.flow import get_datset_file

main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/'

def load_model_weights(prefix):
    model_file =  os.path.join(main_dir, prefix + '_checkpoint.pth.tar')
    model_weigths = torch.load(model_file, map_location=lambda storage, loc: storage)
    return model_weigths

def load_embeddings(embeddings_file):
    with pd.HDFStore(embeddings_file, "r") as fid:    
        embedding_groups = fid['/embedding_groups']
        strains_codes = fid['/strains_codes']
    
    with tables.File(embeddings_file, "r") as fid:
        video_embeddings = fid.get_node('/video_embeddings')[:]
        snps_embeddings = fid.get_node('/snps_embeddings')[:]
    #%%
    em_c = embedding_groups.copy()
    em_c['tot'] = em_c['fin'] - em_c['ini'] + 1
    
    strains_lab = []
    strains_lab_id = []
    for irow, row in em_c.sort_values(by='ini').iterrows():
        ss = row['strain']
        nn = row['fin'] - row['ini'] + 1
        strains_lab += [ss]*nn
    #%%
    dd = strains_codes['strain_id']
    dd.index = strains_codes['strain']
    strains_lab_id = dd[strains_lab].values
    
    snps_embeddings = snps_embeddings[strains_lab_id]
    
    return snps_embeddings, video_embeddings, strains_lab

def get_embeddings_pearson(snps_embeddings, video_embeddings):
    dat = []
    for ii in range(snps_embeddings.shape[1]):
        d = pearsonr(snps_embeddings[:, ii], video_embeddings[:, ii])
        dat.append(d)
        
    pcoeff, pvalue = map(np.array, zip(*dat))
    return pcoeff, pvalue

def get_embeddings_tsne(embeddings, strains_lab, n_samples = 500):
    
    x_inds, Y = [], []
    s_g = pd.DataFrame(np.array(strains_lab), columns=['strain']).groupby('strain')
    for s, inds in s_g.groups.items():
        x_inds += random.sample(list(inds), n_samples)
        Y += [s]*n_samples
    Y = np.array(Y)
    x_inds = np.array(x_inds)
    X = embeddings[x_inds]
    #X_embedded = PCA(n_components=3).fit_transform(X)
    X_embedded = TSNE(n_components=2, verbose=1, n_iter=300).fit_transform(X)

    df = pd.DataFrame(X_embedded[:, :2], columns=['X1', 'X2'])
    df['strain'] = Y

    return df
#%%
def _get_file_parts(embeddings_file):
    bn = os.path.basename(embeddings_file)
    parts = bn.split('_')
    
    n_embeddings = [int(x[1:]) for x in parts if x.startswith('L')][0]
    is_clf = [int(x[3:]) for x in parts if x.startswith('Clf')][0]
    is_ae = [int(x[2:]) for x in parts if x.startswith('AE')][0]
    
    
    return n_embeddings, is_clf, is_ae
   #%% 
    
    
if __name__ == '__main__':
    import glob
    
    fnames = glob.glob(os.path.join(main_dir, '*_embeddings.hdf5'))
    
    all_data = {}
    for embeddings_file in fnames:
        print(embeddings_file)
        snps_emb, video_emb, strains_l = load_embeddings(embeddings_file)
        pcoeff, pvalue = get_embeddings_pearson(snps_emb, video_emb)
        df = get_embeddings_tsne(video_emb, strains_l)
        
        lab = _get_file_parts(embeddings_file)
        
        all_data[lab] = ((snps_emb, video_emb, strains_l), (pcoeff, df))
        
    #%% Plot coeff
    plt.figure()
    legends = {}
    for (n_embeddings, is_clf, is_ae), dat in all_data.items():
        
        nn = 1 if n_embeddings == 32 else 2
        
        lab_str = 'AE={} Clf={}'.format(is_ae,is_clf)
        
        pcoeff = dat[1][0]
        plt.subplot(2,1,nn)
        dd = plt.plot(np.sort(pcoeff), label=lab_str)
    for nn in [1,2]:
        plt.subplot(2,1,nn)
        plt.legend(loc=0)
    #%% Plot tsne s
    all_df = []
    for (n_embeddings, is_clf, is_ae), dat in all_data.items():
        (pcoeff, df) = dat[1]
        
        
        df['n_embeddings'] = n_embeddings
        df['m'] = 'AE={} Clf={}'.format(is_ae,is_clf)
        all_df.append(df)
    
    all_df = pd.concat(all_df)

    sns.lmplot('X1', 'X2', data=all_df, hue='strain', fit_reg=False, col='m', row='n_embeddings')
        
    #%%
    
    
    
    
#    C = np.cov(snps_embeddings_l.T, embeddings.T)
#    plt.figure()
#    plt.imshow(C, interpolation='none')   
    #%%
#    #%%
#    plt.figure()
#    plt.plot(np.sort(pcoeff))
#    #%%
#    dim_sorted = np.argsort(pcoeff)
#    plt.figure()
#    for ii, ind in enumerate(dim_sorted[-9:]):
#        plt.subplot(3,3, ii+1)
#        plt.plot(snps_embeddings_l[:, ind], embeddings[:, ind], '.')
#    
#    
#    #%%
#    #p_inds = np.argsort(np.array(pvalue)*256)
#    #ii = p_inds[40]
#    #plt.figure()
#    #plt.plot(snps_embeddings_l[:, ii], embeddings[:, ii], '.')
#    #%%
##    W0 = model_weigths['state_dict']['snp_mapper.0.weight']
##    W1 = model_weigths['state_dict']['snp_mapper.1.weight']
##    B0 = model_weigths['state_dict']['snp_mapper.0.bias']
##    B1 = model_weigths['state_dict']['snp_mapper.1.bias']
##    
##    m = torch.matmul(W0.t(), W0)
##    m_inv = m.inverse()
##    
#    #%%
#    
#
#    #%%
