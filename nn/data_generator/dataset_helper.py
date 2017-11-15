#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:12 2017

@author: ajaver
"""

import sys
import os

# wild isolates used to test SWDB
SWDB_WILD_ISOLATES = ['JU393', 'ED3054', 'JU394',
                      'N2', 'JU440', 'ED3021', 'ED3017',
                      'JU438', 'JU298', 'JU345', 'RC301',
                      'AQ2947', 'ED3049',
                      'LSJ1', 'JU258', 'MY16',
                      'CB4852', 'CB4856', 'CB4853',
                      ]

# divergent set used for CeNDR
CeNDR_DIVERGENT_SET = ['N2', 'ED3017', 'CX11314', 'LKC34', 'MY16', 'DL238',
                       'JT11398', 'JU775',
                       'JU258', 'MY23', 'EG4725', 'CB4856']

def _h_get_valid_strains(dataset, is_reduced):
    if not is_reduced:
        valid_strains = None
    else:
        if dataset == 'SWDB':
            valid_strains = SWDB_WILD_ISOLATES
        elif dataset == 'CeNDR':
            valid_strains = CeNDR_DIVERGENT_SET
        else:
            raise ValueError('Not valid dataset')
    
    return valid_strains


def _h_get_datset_file(dataset):
    '''
    Get the localization of the dataset file. 
    Only really only used to ease the setup in the training cluster.
    '''
    if sys.platform == 'linux':
        data_dir = os.environ['TMPDIR']
    else:        
        data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/'
        data_dir = os.path.join(data_dir, dataset)
        
    data_file =  os.path.join(data_dir, dataset + '_skel_smoothed.hdf5')
    
    return data_file

