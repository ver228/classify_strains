#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:18:13 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables
import tqdm
import warnings
import matplotlib.pylab as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

def _h_angles(skeletons):
    '''
    Get skeletons angles
    '''
    dd = np.diff(skeletons,axis=1);
    angles = np.arctan2(dd[...,0], dd[...,1])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1);
    
    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    mean_angles = np.unwrap(mean_angles)
    
    return angles, mean_angles

all_pca = np.load('./results/ipca_components.npy')
with open('./results/PCA_errors_SWDB.pkl', 'rb') as fid:
    all_errors = pickle.load(fid)

dd = [(k, v[2].max()) for k,v in all_errors.items()]
strains, _ =  zip(*sorted(dd, key = lambda x: x[1], reverse=True))


set_type = 'SWDB'
main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/{0}/{0}_skel_smoothed.hdf5'.format(set_type)

with pd.HDFStore(main_dir, 'r') as fid:
    skels_ranges = fid['/skeletons_groups']

skel_g = skels_ranges.groupby('strain')
#for i_strain, (strain, dat) in enumerate(skel_g):
    
for strain in strains[:10]:
    pdf = PdfPages('./results/skeletons_errors/{}_skel_errors.pdf'.format(strain))
    
    dat = skel_g.get_group(strain)
    tot_rows = (dat['fin']-dat['ini']+1).sum()
    angs = np.full((tot_rows, 48), np.nan, dtype = np.float32)
    angs_mean = np.full((tot_rows), np.nan, dtype = np.float32)
    
    with tables.File(main_dir, 'r') as fid:
        skel_node = fid.get_node('/skeletons_data')
        is_bad_node = fid.get_node('/is_bad_skeleton')
        
        tot = 0
        dd = tqdm.tqdm(enumerate(dat.iterrows()), total=len(dat), desc = strain)
        for ichuck, (irow, row) in dd:
            
            ini = row['ini']
            fin = row['fin']
            #is_bad_skeleton = is_bad_node[ini:fin+1]>0
            
            skels = skel_node[ini:fin+1, :, :]
            #skels = skels[~is_bad_skeleton]
            #%%
            fig = plt.figure(figsize=(15, 3))
            for nn in range(4,9):
                aa, am = _h_angles(skels)
                pca_r = all_pca[:nn]
                DD = np.dot(aa, pca_r.T)
                DD_a = np.dot(DD, pca_r)    
                
                err = np.abs(DD_a-aa).mean(axis=0)
                
                ind = np.argmax(err)
                segment_l = np.mean(np.linalg.norm(np.diff(skels, axis=1), axis=2), axis=1)
                
                aa_r = DD_a + am[:, None]
                
                xx = np.cos(aa_r)*segment_l[:, None]
                xx = np.hstack((skels[..., 0, 1][:, None],  xx))
                xx = np.cumsum(xx, axis=1) 
                
                yy = np.sin(aa_r)*segment_l[:, None]
                yy = np.hstack((skels[..., 0, 0][:, None],  yy))
                yy = np.cumsum(yy, axis=1)
                
                
                ax = plt.subplot(1,5,nn-3)
                plt.plot(xx[ind], yy[ind], 'o')
                plt.plot(skels[ind, : , 1], skels[ind, : , 0], '.')
                plt.title(nn)
                plt.axis('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
            pdf.savefig(fig)
            plt.close()
        pdf.close()
        #%%
        
#        #%%
#        
#        mean_angles = am
#        angles = aa
#        
#        del_ang = np.hstack((0, np.diff(mean_angles)))
#        del_ss = np.vstack((np.zeros((1,2)), np.diff(skels[:, 0, :], axis=0)))
#        
#        pca_r = all_pca[:8]
#        eigen_vals = np.dot(angles, pca_r.T)
#        
#        VV = np.hstack((del_ss, del_ang[:, None], eigen_vals))
#        
#        plt.figure()
#        plt.imshow(VV.T, aspect='auto', interpolation='none')
#        
#        if ichuck > 10:
#            break
        

