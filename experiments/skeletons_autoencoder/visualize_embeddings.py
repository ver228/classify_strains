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
from classify.flow import get_datset_file, get_valid_strains


def load_model_weights(prefix):
    model_file =  os.path.join(main_dir, prefix + '_checkpoint.pth.tar')
    model_weigths = torch.load(model_file, map_location=lambda storage, loc: storage)
    return model_weigths

def load_embeddings(embeddings_file, is_reduced=True, valid_set = 'test'):
    with pd.HDFStore(embeddings_file, "r") as fid:    
        embedding_groups = fid['/embedding_groups']
        strains_codes = fid['/strains_codes']
    
    valid_strain = get_valid_strains('CeNDR', is_reduced=is_reduced)
    embedding_groups = embedding_groups[embedding_groups['strain'].isin(valid_strain)]
    
    with tables.File(embeddings_file, "r") as fid:
        if valid_set and valid_set is not None:
            valid_inds = fid.get_node('/index_groups/' + valid_set)[:]
            embedding_groups = embedding_groups[embedding_groups['skel_group_id'].isin(valid_inds)]
    
        snps_embeddings = fid.get_node('/snps_embeddings')[:]
        video_embeddings = []
        for irow, row in embedding_groups.iterrows():
            ini, fin = row['ini'], row['fin']
            dd = fid.get_node('/video_embeddings')[ini:fin+1]
            video_embeddings.append(dd)
        video_embeddings = np.concatenate(video_embeddings)
    
    em_c = embedding_groups.copy()
    em_c['tot'] = em_c['fin'] - em_c['ini'] + 1
    
    strains_lab = []
    strains_lab_id = []
    for irow, row in em_c.sort_values(by='ini').iterrows():
        ss = row['strain']
        nn = row['fin'] - row['ini'] + 1
        strains_lab += [ss]*nn
    
    dd = strains_codes['strain_id']
    dd.index = strains_codes['strain']
    strains_lab_id = dd[strains_lab].values
    
    snps_embeddings = snps_embeddings[strains_lab_id]
    
    assert snps_embeddings.shape[0] == video_embeddings.shape[0]
    assert len(strains_lab) == video_embeddings.shape[0]
    
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
    def isfloat(value):
      try:
        float(value)
        return True
      except:
        return False
    
    bn = os.path.basename(embeddings_file)
    parts = bn.split('_')
    
    n_embeddings = [int(x[1:]) for x in parts if x.startswith('L')][0]
    is_clf = [int(x[3:]) for x in parts if x.startswith('Clf')][0]
    is_ae = [int(x[2:]) for x in parts if x.startswith('AE')][0]
    mix_emb = [float(x[3:]) for x in parts if x.startswith('Emb') and isfloat(x[3:])][0]
    is_reduced = any(x=='R' for x in parts)
    model_name = parts[0]
    return model_name, n_embeddings, is_clf, is_ae, mix_emb, is_reduced
   #%% 
    

if __name__ == '__main__':
    import glob
    import multiprocessing as mp
    
    def _process_row(embeddings_file):
        print(embeddings_file)
        snps_emb, video_emb, strains_l = load_embeddings(embeddings_file)
        pcoeff, pvalue = get_embeddings_pearson(snps_emb, video_emb)
        df = get_embeddings_tsne(video_emb, strains_l)
        
        lab = _get_file_parts(embeddings_file)
        
        val = ((snps_emb, video_emb, strains_l), (pcoeff, df))
        return lab, val
    
    #main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/vae_w_embeddings/'    
    #main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/20171129_reduced'
    #main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/20171130_full'
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/20171130_reduced'
    
    fnames = glob.glob(os.path.join(main_dir, '*_embeddings.hdf5'))
    
    p = mp.Pool()
    results = p.map(_process_row, fnames)
    all_data = {k:v for k,v in results}
    
    #%% Plot coeff
    for is_reduced in [True, False]:
        valid_k = [k for k in all_data.keys() if k[-1] == is_reduced]
        valid_k = sorted(valid_k)
        if not valid_k:
            continue
        
        available_n_embeddings = sorted(list(set(x[1] for x in valid_k)))
        subplot_n = {k:ii+1 for ii,k in enumerate(available_n_embeddings)}
        
        #%%
        fig = plt.figure()
        
        legends = {}
        for k in valid_k:
            model_name, n_embeddings, is_clf, is_ae, mix_val, _ = k
            dat = all_data[k]
            lab_str = 'AE={} Clf={} Mix={}'.format(is_ae,is_clf, mix_val)
            
            pcoeff = dat[1][0]
            plt.subplot(2,1,subplot_n[n_embeddings])
            dd = plt.plot(np.sort(pcoeff), label=lab_str)
        
        
        plt.suptitle('is_reduced={}'.format(is_reduced))
        for n_embeddings in subplot_n:
            nn = subplot_n[n_embeddings]
            plt.subplot(2,1,nn)
            plt.legend(loc=0)
            plt.xlim((-1, n_embeddings+1))
        
        plt.xlabel('Embedding Index')
        #plt.ylabel('SNP-Angles Pearson Coeff')
        fig.text(0.04, 0.5, 'SNP-Angles Pearson Coeff', va='center', rotation='vertical')
        
        fname = 'correlations_{}.png'.format('reduced' if is_reduced else 'full')
        fname = os.path.join(main_dir, fname)
        fig.savefig(fname)
        #%%
        
        all_df = []
        for k in valid_k:
            model_name, n_embeddings, is_clf, is_ae, mix_val, _ = k
            dat = all_data[k]
            (pcoeff, df) = dat[1]
            
            df['n_embeddings'] = n_embeddings
            df['m'] = 'AE={} Clf={} Mix={}'.format(is_ae, is_clf, mix_val)
            all_df.append(df)
        
        all_df = pd.concat(all_df)
        #%%
        fig = sns.lmplot('X1', 'X2', data=all_df, hue='strain', fit_reg=False, col='m', row='n_embeddings')
        #plt.suptitle('is_reduced={}'.format(is_reduced))
        
        fname = 'tSNE_{}.png'.format('reduced' if is_reduced else 'full')
        fname = os.path.join(main_dir, fname)
        fig.savefig(fname)
        