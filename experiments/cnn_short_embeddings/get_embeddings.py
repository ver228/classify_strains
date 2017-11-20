#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:57:25 2017

@author: ajaver
"""
import sys
import os
import pandas as pd
import torch
import tqdm
from torch import nn
import torch.nn.functional as F
    
src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.models.resnet import ResNetS, Bottleneck
from classify.flow import SkeletonsFlowFull, get_valid_strains, get_datset_file

class HeadlessModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(original_model.children())[:-1]
        )
    def forward(self, x):
        x = self.features(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

if __name__ == '__main__':
    import tables
    import numpy as np
    
    models_path = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/model_20171103/'
    model_name = 'resnet50_R_CeNDR_ang__S10_F0.04_20171104_182812.pth.tar'
    dataset = 'CeNDR'
    n_classes = 197
    
    
    fname = os.path.join(models_path, model_name)
    model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes = n_classes)
    checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    
    data_file = get_datset_file(dataset)
    valid_strains = get_valid_strains(dataset, is_reduced=True)
    
    dd = model_name.split('_')
    sample_size_seconds = [float(x[1:]) for x in dd if x.startswith('S')][0]
    sample_frequency_s = [float(x[1:]) for x in dd if x.startswith('F')][0]
    
    gen = SkeletonsFlowFull(
                          n_batch = 32, 
                          data_file = data_file,
                          sample_size_seconds = sample_size_seconds, 
                          sample_frequency_s = sample_frequency_s,
                          valid_strains = valid_strains,
                          is_torch = True,
                          label_type = 'row_id'
                          )
    results = []
    
    model_headless = HeadlessModel(original_model=model)
    for input_v, row_ids in tqdm.tqdm(gen):
        embedding = model_headless.forward(input_v)
        results.append((row_ids.data.numpy(), embedding.data.numpy()))
    
    labels, embeddings = zip(*results)
    #%%
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    #%%
    df_g = pd.DataFrame(labels.T, columns=['row_id']).groupby('row_id').groups
    #%%
    with pd.HDFStore(data_file, 'r') as fid:
        skeletons_groups = fid['/skeletons_groups']
    
    #%%
    embedding_groups = []
    for irow, row in skeletons_groups.iterrows():
        if irow in df_g:
            row_n = row.copy()
            row_n['ini'] = df_g[irow].min()
            row_n['fin'] = df_g[irow].max()
            row_n['skel_group_id'] = irow
            embedding_groups.append(row_n)
    embedding_groups = pd.DataFrame(embedding_groups)
    
    #%%
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    new_file = data_file.replace('_skel_smoothed.hdf5', '_embedings.hdf5')
    fields2copy = ['experiments_data', 'snps_data', 'strains_codes']
    
    with tables.File(data_file, 'r') as fid_old, \
            tables.File(new_file, "w") as fid_new:
        
        for field in fields2copy:
            tab = fid_old.get_node('/' +field)[:]        
            fid_new.create_table('/', 
                             field,
                             obj=tab,
                             filters=TABLE_FILTERS)
        gg = fid_new.create_group('/', 'index_groups')
        for field in ['train', 'test', 'val']:
            tab = fid_old.get_node('/index_groups/' +field)[:]        
            fid_new.create_array('/index_groups', 
                             field,
                             obj=tab)
    #%%
    
    with tables.File(new_file, "r+") as fid:
        table_type = np.dtype([('experiment_id', np.int32),
                               ('worm_index', np.int32),
                          ('strain', 'S10'),
                          ('ini_time_aprox', np.float32),
                          ('ini', np.int32),
                          ('fin', np.int32),
                          ('skel_group_id', np.int32)
                          ])
        em = embedding_groups.to_records(index=False).astype(table_type)
        fid.create_table('/', 
                             'embedding_groups',
                             obj=em,
                             filters=TABLE_FILTERS)
        fid.create_carray('/', 
                             'embeddings',
                             obj=embeddings,
                             filters=TABLE_FILTERS)
    
