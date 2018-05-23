#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:17:29 2018

@author: avelinojaver
"""
import os
import sys

_root_dirs = {
        'loc' : '/Volumes/rescomp1/data/WormData/',
        'ox' : '/well/rittscher/users/avelino/WormData/'
        }

_set_dirs = {
        'skels' : 'screenings/CeNDR/Results',
        'AE_emb_20180206' : 'experiments/classify_strains/autoencoders/CeNDR_ROIs_embeddings/AE2D__snippet5_trainL32_20180206_123614_epoch107'
        }

def get_path(set_type, root = None):
    if root is None:
        root = 'loc' if sys.platform == 'darwin' else 'ox'
    
    data_dir = os.path.join(_root_dirs[root], _set_dirs[set_type])
    results_dir = os.path.join(_root_dirs[root], 'experiments/classify/')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return data_dir, results_dir