#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:17:29 2018

@author: avelinojaver
"""
import os
import sys

_root_dirs = {
        'osx' : '/Volumes/rescomp1/data/WormData/',
        'centos_oxford' : '/well/rittscher/users/avelino/WormData/',
        'loc' : '/Users/avelinojaver/Documents/Data/',
        'tmp' : '/tmp/'
        }

def get_path(emb_set, platform = None, is_tmp = True):
    
    if platform is None:
        platform = 'osx' if sys.platform == 'darwin' else 'centos_oxford'
    
    root = _root_dirs[platform]
    
    
    fname = root + 'experiments/classify_strains/CeNDR_{}.hdf5'.format(emb_set)
    results_dir = os.path.join(root, 'experiments/classify_strains/results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return fname, results_dir