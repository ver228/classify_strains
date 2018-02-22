#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:11:04 2018

@author: ajaver
"""

from models import AE2D
from flow import shift_and_normalize

import os
import shutil
import torch
import glob
import pandas as pd
import tables
import tqdm
import numpy as np

is_cuda = True
roi_dir = '/data/ajaver/CeNDR_ROIs'
save_dir_root = '/data/ajaver/CeNDR_ROIs_embeddings'
model_path_root = '/home/ajaver@cscdom.csc.mrc.ac.uk/Github/classify_strains/results'

assert roi_dir[-1] != os.path.sep
assert save_dir_root[-1] != os.path.sep
assert model_path_root[-1] != os.path.sep

TABLE_FILTERS = tables.Filters(
    complevel=5,
    complib='zlib',
    shuffle=True,
    fletcher32=True)

if __name__ == '__main__':
    
    model_name = 'AE2D__snippet5_trainL32_20180206_123614'
    embedding_size = 32
    model_file = os.path.join(model_path_root, model_name, 'checkpoint.pth.tar')
    
    #load model
    model =  AE2D(embedding_size)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    if is_cuda:
        model.cuda()
    model.eval()
    
    #set directory where the embeddings are going to be saved
    save_dir = os.path.join(save_dir_root, '{}_epoch{}'.format(model_name, checkpoint['epoch']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shutil.copy(model_file, save_dir)
    
    fnames = glob.glob(os.path.join(roi_dir, '**', '*ROIs.hdf5'), recursive=True)
    #%%
    seq_size = 1024
    
    for rois_file in fnames:
        embeddings_file = rois_file.replace('_ROIs.hdf5', '_embeddings.hdf5').replace(roi_dir, save_dir)
        dname = os.path.dirname(embeddings_file)
        if not os.path.exists(dname):
            os.makedirs(dname)
        
        with pd.HDFStore(rois_file, 'r') as fid:
            trajectories_data = fid['/trajectories_data']
        
        
        
        tot_rows = len(trajectories_data)
        with tables.File(embeddings_file, 'w') as fid, tables.File(rois_file, 'r') as fid_roi:
            tab = fid.create_table('/',
                        "trajectories_data",
                        trajectories_data.to_records(index=False),
                        filters = TABLE_FILTERS)
            tab._v_attrs['has_finished'] = 0
            
            
            embs_array = fid.create_carray('/', 
                    'embeddings',
                    atom = tables.Float32Atom(),
                    shape = (tot_rows, embedding_size),
                    filters = TABLE_FILTERS
                    )
            
            
            masks = fid_roi.get_node('/mask')
            for ii in tqdm.tqdm(range(0, tot_rows, seq_size)):
                rois = masks[ii:ii+seq_size]
                rois = shift_and_normalize(rois.astype(np.float32)) + 0.5
                #add extra dimension (channel)
                S = torch.from_numpy(rois[None, :, None, ...])
                if is_cuda:
                    S = S.cuda()
                
                S = torch.autograd.Variable(S)
                emb_g = model.encoder(S)
                emb_c = emb_g.cpu()
                
                embs_array[ii:ii+seq_size] = emb_c.squeeze().data.numpy()
                
                del S
                del emb_g
                del emb_c
            
            tab._v_attrs['has_finished'] = 1
            
    
    