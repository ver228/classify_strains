#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:59:28 2017

@author: ajaver
"""

import os
import glob
import shutil

#dst_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/ae_w_embeddings'
#log_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/skeletons_autoencoder'
log_dir = '/work/ajaver/classify_strains/results/skeletons_autoencoder/'
dst_dir = '/work/ajaver/classify_strains/trained_models/ae_w_embeddings/'

#log_dir = '/work/ajaver/classify_strains/results/skeletons_vae/'
#dst_dir = '/work/ajaver/classify_strains/trained_models/vae_w_embeddings/'


fnames = glob.glob(os.path.join(log_dir, '**', 'checkpoint*'), recursive=True)


for fname in fnames:
    parts = fname.split(os.sep)
    new_name = os.path.join(dst_dir, parts[-2] + '_' + parts[-1])
    shutil.copyfile(fname, new_name)