#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:41 2017

@author: ajaver
"""
import os
from collections import OrderedDict
from itertools import product
base_file = '''
#!/bin/sh
#PBS -l walltime={time_str}
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

#PBS -q gpgpu
## This tells the batch manager to enqueue the job in the general gpgpu queue.

module load anaconda3
module load cuda
## This job requires CUDA support.
source activate tierpsy

## copy temporary files
cp $WORK/classify_strains/train_set/{train_file} $TMPDIR/

{cmd_str}
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
'''

if __name__ == '__main__':
    save_dir = './cmd_scripts/grid_011117'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    time_str = '24:00:00'
    main_file = '$HOME/classify_strains/nn/train.py'
    
    params_grid = OrderedDict(
    model_type = ['resnet50', 'gru', 'lstm'],
    is_CeNDR = [True, False],
    is_angle = [True],
    is_reduced = [True, False],
    sample_size_frames_s = [20, 90],
    sample_frequency_s = [1/10.],
    n_epochs = [300]
    )
    
    short_add = dict(
    model_type = lambda x : x,
    sample_size_frames_s = lambda x : 'S{}'.format(x),
    sample_frequency_s = lambda x : 'F{:.2}'.format(x),
    n_epochs = lambda x:None,
    is_angle = lambda x : 'ang' if x else 'xy',
    is_CeNDR = lambda x : 'CeNDR' if x else 'SWDB',
    is_reduced = lambda x : 'R' if x else None,
    )
    
    args_comb = [[(k, x) for x in v] for k,v in params_grid.items()]
    
    for args in product(*args_comb):
        args_d = ' '.join(['--{} {}'.format(*d) for d in args])
        cmd_str = 'python {} {}'.format(main_file, args_d)
        
        is_CeNDR = [y for x,y in args if x == 'is_CeNDR'][0]
        if is_CeNDR:
            train_file = 'CeNDR_skel_smoothed.hdf5'
        else:
            train_file = 'SWDB_skel_smoothed.hdf5'
        
        f_content = base_file.format(time_str=time_str, 
                              cmd_str=cmd_str, 
                              train_file=train_file
                              )
        f_name = [short_add[k](v) for k,v in args]
        f_name = '_'.join([x for x in f_name if x is not None]) + '.sh'
        f_name = os.path.join(save_dir, f_name)
        
        with open(f_name, 'w') as fid:
            fid.write(f_content)
        