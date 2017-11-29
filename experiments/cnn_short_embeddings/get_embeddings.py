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
import numpy as np
import tables

#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.flow import SkeletonsFlowFull, get_valid_strains, get_datset_file
import classify.models.model_w_embedding as models


if __name__ == '__main__':
    dname = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models'
    
    #model_file = 'resnet18_w_emb_R_L256_l2_0.1_20171126_010058_best.pth.tar'
    #props = dict(
    #        model_name = 'resnet18_w_emb',
    #        is_residual = True,
    #        embedding_size = 256
    #        )
    
    model_file = 'simple_w_emb_R_L256_l2_0.01_20171126_010327_best.pth.tar'
    props = dict(
            model_name = 'simple_w_emb',
            is_residual = True,
            embedding_size = 256
            )
    
    
    dataset = 'CeNDR'
    valid_strains = get_valid_strains(dataset, is_reduced=True)
    data_file = get_datset_file(dataset)
    gen = SkeletonsFlowFull(
                          n_batch = 32, 
                          data_file = data_file,
                          sample_size_seconds = 10, 
                          sample_frequency_s = 0.04,
                          valid_strains = valid_strains,
                          label_type = 'row_id',
                          is_return_snps = False,
                          transform_type = 'angles'
                          )
    
    get_model_func = getattr(models, props['model_name'])
    model = get_model_func(gen, props['embedding_size'])
    
    fname = os.path.join(dname, model_file)
    checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    results = []
    for ii, (input_v, row_ids) in enumerate(tqdm.tqdm(gen)):
        video_embedding = model.video_model(input_v)
        pred = model.classification(video_embedding).max(1)[1]
        dat = [x.data.numpy() for x in (row_ids, video_embedding, pred)]    
        results.append(dat)
        
    #%%
    row_ids, embeddings, predictions = map(np.concatenate, zip(*results))
    
    df_g = pd.DataFrame(row_ids.T, columns=['row_id']).groupby('row_id').groups
    embedding_groups = []
    for irow, row in gen.skeletons_ranges.iterrows():
        if irow in df_g:
            row_n = row[['experiment_id',  'worm_index', 'strain', 'strain_id']].copy()
            row_n['ini'] = df_g[irow].min()
            row_n['fin'] = df_g[irow].max()
            row_n['skel_group_id'] = irow
            embedding_groups.append(row_n)
    embedding_groups = pd.DataFrame(embedding_groups)
    #%%
    snps_embeddings = np.full((gen.n_classes, props['embedding_size']), np.nan, dtype=np.float32)
    
    for strain_id in gen.strain_ids:
        strain = gen.strain_codes.loc[strain_id, 'strain']
        snps = gen.snps_data[strain].values.T.astype(np.float32)
        
        snps = torch.from_numpy(snps).float()
        snps = torch.autograd.Variable(snps)
        snps_embedding = model.snp_mapper(snps)
        snps_embeddings[strain_id] = snps_embedding.data.numpy()
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
                          ('ini', np.int32),
                          ('fin', np.int32),
                          ('skel_group_id', np.int32)
                          ])
        em = embedding_groups[list(table_type.names)].to_records(index=False).astype(table_type)
        fid.create_table('/', 
                             'embedding_groups',
                             obj=em,
                             filters=TABLE_FILTERS)
        fid.create_carray('/', 
                             'video_embeddings',
                             obj=embeddings,
                             filters=TABLE_FILTERS)
        fid.create_carray('/', 
                             'predicted_strain_id',
                             obj=predictions,
                             filters=TABLE_FILTERS)
        fid.create_carray('/', 
                             'snps_embeddings',
                             obj=snps_embeddings,
                             filters=TABLE_FILTERS)