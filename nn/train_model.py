#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:50:59 2017

@author: ajaver
"""
import torch
import sys
import os
import math

from models import ResNetS, Bottleneck
from skeletons_flow import SkeletonsFlow, CeNDR_DIVERGENT_SET, SWDB_WILD_ISOLATES

is_cuda = torch.cuda.is_available()

if sys.platform == 'linux':
    LOG_DIR = '/work/ajaver/classify_strains/results'
    DATA_DIR = os.environ['TMPDIR']
else:        
    LOG_DIR = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
    DATA_DIR = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/'
    


sample_size_frames_s_dflt = 90.
sample_frequency_s_dflt = 1/10
n_batch_base = 32
sample_size_frames_s = sample_size_frames_s_dflt
sample_frequency_s = sample_frequency_s_dflt
valid_strains = None

is_angle = True
is_CeNDR = True
is_reduced = True

if is_CeNDR:
    data_file =  os.path.join(DATA_DIR, 'CeNDR', 'CeNDR_skel_smoothed.hdf5')
    bn_prefix = 'CeNDR_'
else:
    data_file = os.path.join(DATA_DIR, 'SWDB', 'SWDB_skel_smoothed.hdf5')
    bn_prefix = 'SWDB_'
    
if is_reduced:
    if is_CeNDR:
        valid_strains = CeNDR_DIVERGENT_SET
    else:
        valid_strains = SWDB_WILD_ISOLATES
    bn_prefix = 'R_' + bn_prefix

if is_angle:
    bn_prefix += 'ang_'
else:
    bn_prefix += 'xy_'

factor = sample_size_frames_s/sample_size_frames_s_dflt
factor *= (sample_frequency_s_dflt/sample_frequency_s)
n_batch = max(int(math.floor(n_batch_base/factor)), 1)


train_generator = SkeletonsFlow(data_file = data_file, 
                       n_batch = 32, 
                       set_type = 'train',
                       valid_strains = valid_strains,
                       is_angle = is_angle
                       )
test_generator = SkeletonsFlow(data_file = data_file, 
                       n_batch = 32, 
                       set_type = 'test',
                       valid_strains = valid_strains,
                       is_angle = is_angle
                       )

# to prevent opencv from initializing CUDA in workers
#torch.randn(8).cuda()
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes=20)
if is_cuda:
    model.cuda()


lr=1e-3
torch.optim.Adam(model.parameters(), lr)


X,Y = next(train_generator)


