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
from scipy.stats.stats import pearsonr
import os
import sys
import torch
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation

src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src')
sys.path.append(src_dir)
from classify.flow import get_valid_strains

import models
            
def load_model_weights(prefix):
    model_file =  os.path.join(main_dir, prefix + '_checkpoint.pth.tar')
    model_weigths = torch.load(model_file, map_location=lambda storage, loc: storage)
    return model_weigths

def load_embeddings(embeddings_file, is_reduced=True, valid_set = 'test'):
    with pd.HDFStore(embeddings_file, "r") as fid:    
        embedding_groups = fid['/embedding_groups']
        strains_codes = fid['/strains_codes']
    
    valid_strain = get_valid_strains('CeNDR', is_reduced=is_reduced)
    if valid_strain is not None:
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
    
    
    divergent_set = get_valid_strains('CeNDR', is_reduced=True)
    divergent_set = sorted(divergent_set)
    
    def _process_row(embeddings_file, is_reduced=False, valid_set = 'test'):
        print(embeddings_file)
        snps_emb, video_emb, strains_l = load_embeddings(embeddings_file, 
                                                         is_reduced=is_reduced, 
                                                         valid_set=valid_set)
        pcoeff, pvalue = get_embeddings_pearson(snps_emb, video_emb)
        
        lab = _get_file_parts(embeddings_file)
        
        val = ((snps_emb, video_emb, strains_l), (pcoeff))
        return lab, val, embeddings_file
        
    #main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/20171129_reduced'
    #main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings/20171130_full'
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/vae_w_embeddings/'

    fnames = glob.glob(os.path.join(main_dir, '*_embeddings.hdf5'))
    
    p = mp.Pool()
    results = p.map(_process_row, fnames)
    #%%
    all_data = {k:(v,f) for k, v, f in results}
            
    #%% Plot coeff
    for is_reduced in [False]:#[True, False]:
        valid_k = [k for k in all_data.keys() if k[-1] == is_reduced]
        valid_k = [k for k in all_data.keys() if k[2] == True]
        valid_k = [k for k in all_data.keys() if k[1] == 64]
        
        if not valid_k:
            continue
        
        available_n_embeddings = sorted(list(set(x[1] for x in valid_k)))
        subplot_n = {k:ii+1 for ii,k in enumerate(available_n_embeddings)}
        
        plt.figure()
        legends = {}
        for k in valid_k:
            model_name, n_embeddings, is_clf, is_ae, mix_val, _ = k
            dat, embeddings_file = all_data[k]
            (snps_emb, video_emb, strains_l), (pcoeff) = dat
            lab_str = 'AE={} Clf={} Mix={}'.format(is_ae,is_clf, mix_val)
            
            plt.subplot(2,1,subplot_n[n_embeddings])
            dd = plt.plot(np.sort(pcoeff), label=lab_str)
       
        plt.suptitle('is_reduced={}'.format(is_reduced))
        for n_embeddings in subplot_n:
            nn = subplot_n[n_embeddings]
            plt.subplot(2,1,nn)
            plt.legend(loc=0)
            plt.xlim((-1, n_embeddings+1))
        
        
        for k in valid_k:
            model_name, n_embeddings, is_clf, is_ae, mix_val, _ = k
            dat, embeddings_file = all_data[k]
            (snps_emb, video_emb, strains_l), (pcoeff) = dat
            g_s  = pd.DataFrame(strains_l, columns=['strain']).groupby('strain').groups
            
            
            inds = np.argsort(pcoeff)
            #%%
            v_mean = []
            snps = []
            strains = []
            v_std = []
            for ss, ind in g_s.items():
                v_mean.append(np.median(video_emb[ind], axis=0))
                v_std.append(np.std(video_emb[ind], axis=0))
                #v_mean.append(video_emb[ind[0]])
                snps.append(snps_emb[ind[0]])
                strains.append(ss)
            v_mean = np.vstack(v_mean)
            v_std = np.vstack(v_std)
            snps = np.vstack(snps)
            strains = np.array(strains)
            
            #%%
#            lab_str = 'L{}_AE={}_Clf={}_Mix={}'.format(n_embeddings, is_ae,is_clf, mix_val)
#            
#            save_name = 'JoinPlots_{}.pdf'.format(lab_str)
#            if is_reduced:
#                save_name = 'R_' + save_name
#            
#            save_name = os.path.join(os.path.dirname(main_dir), save_name)
#            
#            with PdfPages(save_name) as pdf:    
#                for nn in inds[::-1]:
#                    print(nn)
#                    
#                    sns.jointplot(x=v_mean[:,nn].T, y=snps[:,nn].T, ratio=2, size=5);
#                    pdf.savefig() 
#                    plt.close()
            #%%
#            get_model_func = getattr(models, model_name)
#            
#            n_classes = 198
#            n_snps = 37851
#            model = get_model_func(n_classes, 
#                               n_snps, 
#                               n_embeddings)
#            
#            model_path = embeddings_file.replace('_embeddings.hdf5', '_checkpoint.pth.tar')
#            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#            model.load_state_dict(checkpoint['state_dict'])
#            model.eval()
#            
#            #%%
#            x = v_mean
#            x = torch.autograd.Variable(torch.from_numpy(x))
#            mm = model.video_decoder(x)
#            mm = mm.data.squeeze().numpy()
#            #%%
#            for ii, ss in enumerate(divergent_set):
#                angles = np.squeeze(mm[strains == ss])
#                ske_x = np.cos(angles)
#                ske_x = np.cumsum(ske_x, axis=1) 
#                
#                ske_y = np.sin(angles)
#                ske_y = np.cumsum(ske_y, axis=1) 
#                
#                dpi = 100
#                fig, ax = plt.subplots()
#                ln, = plt.plot([], [], 'ro', animated=True)
#                
#                def init():
#                    ax.set_xlim(-1, 51)
#                    ax.set_ylim(-25, 25)
#                    return ln,
#                    
#                def update(n):
#                    ln.set_data(ske_x[n], ske_y[n])
#                    return ln,
#                
#                
#                ani = animation.FuncAnimation(fig, update, range(128), interval = 40, init_func=init, blit=True)
#                plt.show()
#                
#                writer = animation.writers['ffmpeg'](fps = 25)
#                dd = os.path.basename(model_path).replace('_checkpoint.pth.tar', '')
#                ani.save('{}_{}.mp4'.format(ss, dd),writer=writer,dpi=dpi)
#                break
                #%%
                #plt.figure()
                #plt.imshow(angles.T, interpolation='none', cmap='jet')
                #plt.grid('off')
            #%%
            for dd_s in divergent_set:
                plt.figure()
                inds = np.argsort(pcoeff)
                for ii in range(64):
                    plt.subplot(8,8, ii+1)
                    for ss in ['CB4856', dd_s]:
                        good = np.array(strains_l) == ss
                        cc, bins = np.histogram(video_emb[good][inds[ii]])
                        plt.plot(bins[:-1], cc)
                    plt.title(inds[ii])
                plt.suptitle(dd_s)
                
            #%%
            #plt.plot(np.sort(v_mean[:, 0]))
            good = np.in1d(strains, np.array(divergent_set))
            plt.figure()
            for ii in range(64):
                plt.subplot(8,8, ii+1)
                plt.plot(np.sort(v_mean[:, ii]))
            