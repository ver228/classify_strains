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

is_cuda = torch.cuda.is_available()
import models


#Be sure to use abspath linux does not give the path if one uses __file__
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(_BASEDIR, os.pardir, os.pardir, 'src')
sys.path.append(src_dir)

from classify.flow import SkeletonsFlowFull, get_valid_strains, get_datset_file

def save_embeddings(model_path):
    bn = os.path.basename(model_path)
    parts = bn.split('_')    
    model_name = parts[0]
    embedding_size = [int(x[1:]) for x in parts if x.startswith('L')][0]
    is_reduced = False #get all the strains, even if it was not trained with them
    #is_reduced = any(x=='R' for x in parts)

    dataset = 'CeNDR'
    valid_strains = get_valid_strains(dataset, is_reduced = is_reduced)
    data_file = get_datset_file(dataset)

    gen = SkeletonsFlowFull(
                            label_type = 'row_id',
                            n_batch = 128, 
                            data_file = data_file,
                            sample_size_seconds = 10,
                            sample_frequency_s = 0.04,
                            is_return_snps = False,
                            is_autoencoder = False,
                            valid_strains = valid_strains,
                            is_cuda = is_cuda
                          )

    embeddings_file = model_path.replace('_checkpoint.pth.tar', '_embeddings.hdf5')
    
    
    get_model_func = getattr(models, model_name)
    model = get_model_func(gen.n_classes, 
                           gen.n_snps, 
                           embedding_size)
    model.load_from_file(model_path)
    
    if is_cuda:
        model = model.cuda()
    
    #%% Get video embeddings
    results = []
    for ii, (input_v, row_ids) in enumerate(tqdm.tqdm(gen)):
        video_embedding = model.video_encoder(input_v)
        pred = model.classification(video_embedding).max(1)[1]
        dat = [x.data.cpu().numpy() for x in (row_ids, video_embedding, pred)]    
        results.append(dat)
        
    row_ids, embeddings, predictions = map(np.concatenate, zip(*results))
    #%% group video embeddings by trajectory
    df_g = pd.DataFrame(row_ids.T, columns=['row_id']).groupby('row_id').groups
    embedding_groups = []
    for irow, row in gen.skeletons_ranges.iterrows():
        if irow in df_g:
            row_n = row[['experiment_id',  'worm_index', 'strain', 'strain_id']].copy()
            row_n['ini'] = df_g[irow].min()
            row_n['fin'] = df_g[irow].max()
            row_n['skel_group_id'] = irow
            embedding_groups.append(row_n)
            
    #%% Get snps embeddings
    embedding_groups = pd.DataFrame(embedding_groups)
    
    
    snps_embeddings = np.full((gen.n_classes, embedding_size), np.nan, dtype=np.float32)
    
    for strain_id in gen.strain_ids:
        strain = gen.strain_codes.loc[strain_id, 'strain']
        snps = gen.snps_data[strain].values.T.astype(np.float32)
        
        snps = torch.from_numpy(snps).float()
        if is_cuda:
            snps = snps.cuda()
        
        snps = torch.autograd.Variable(snps)
        snps_embedding = model.snp_mapper(snps)
        snps_embeddings[strain_id] = snps_embedding.data.cpu().numpy()
    #%% save embeddings into file
    TABLE_FILTERS = tables.Filters(
        complevel=5,
        complib='zlib',
        shuffle=True,
        fletcher32=True)
    
    
    fields2copy = ['experiments_data', 'snps_data', 'strains_codes']
    
    with tables.File(gen.data_file, 'r') as fid_old, \
            tables.File(embeddings_file, "w") as fid_new:
        
        for field in fields2copy:
            tab = fid_old.get_node('/' +field)[:]        
            fid_new.create_table('/', 
                             field,
                             obj=tab,
                             filters=TABLE_FILTERS)
        
        fid_new.create_group('/', 'index_groups')
        for field in ['train', 'test', 'val']:
            tab = fid_old.get_node('/index_groups/' +field)[:]    
            fid_new.create_array('/index_groups', 
                             field,
                             obj=tab)
    
    with tables.File(embeddings_file, "r+") as fid:
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

if __name__ == '__main__':
    import glob
    
    if sys.platform == 'linux':
        main_dir = os.path.join(os.environ['TMPDIR'], 'vae_w_embeddings')
    else:        
        main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/vae_w_embeddings/'
    
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L32_Clf1_Emb0.1_AE1_20171129_001223_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L256_Clf1_Emb0.1_AE1_20171129_001337_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L32_Clf0_Emb0.1_AE1_20171129_001219_checkpoint.pth.tar')
    #model_path = os.path.join(main_dir, 'EmbeddingAEModel_R_L256_Clf0_Emb0.1_AE1_20171129_001213_checkpoint.pth.tar')
    
    model_paths = glob.glob(os.path.join(main_dir, '*checkpoint.pth.tar'))
    for fname in model_paths:
        save_embeddings(fname)
    