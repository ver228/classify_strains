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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import os
import sys
src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src')
sys.path.append(src_dir)
from classify.flow import get_datset_file

if __name__ == '__main__':
    data_file = get_datset_file('CeNDR')
    embeddings_file = data_file.replace('_skel_smoothed.hdf5', '_embedings.hdf5')
    
    with pd.HDFStore(embeddings_file, "r") as fid:    
        embedding_groups = fid['/embedding_groups']
    with tables.File(embeddings_file, "r") as fid:
        embeddings = fid.get_node('/embeddings')[:]
    #%%
    em_c = embedding_groups.copy()
    em_c['tot'] = em_c['fin'] - em_c['ini'] + 1
    s_g = em_c.groupby('strain')
    
    #dd = s_g.agg({'tot':'sum'})
    tot = embeddings.shape[0]
    strains_lab = []
    for irow, row in em_c.sort_values(by='ini').iterrows():
        ss = row['strain']
        nn = row['fin'] - row['ini'] + 1
        strains_lab += [ss]*nn
    #%%
    import random
    x_inds, Y = [], []
    n_samples = 2000
    s_g = pd.DataFrame(np.array(strains_lab), columns=['strain']).groupby('strain')
    for s, inds in s_g.groups.items():
        x_inds += random.sample(list(inds), n_samples)
        Y += [s]*n_samples
    Y = np.array(Y)
    x_inds = np.array(x_inds)
    X = embeddings[x_inds]
    #%%
    #X_embedded = PCA(n_components=3).fit_transform(X)
    #
    
    
#    x = np.exp(X - np.max(X, axis=1)[:, None])
#    e_x = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
#%%
    import seaborn as sns
    
    for p in [40]:#range(5, 51, 5):
        print(p)
        X_embedded = TSNE(n_components=2, verbose=1, n_iter=300).fit_transform(X)
        df = pd.DataFrame(X_embedded, columns=['X1', 'X2'])
        df['strain'] = Y
        
    
        plt.figure()    
        sns.lmplot('X1', 'X2', data=df, hue='strain', fit_reg=False)
    
    #plt.plot(X_embedded[:,0], X_embedded[:,1])