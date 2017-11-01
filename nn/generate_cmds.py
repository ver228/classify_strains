#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:41 2017

@author: ajaver
"""
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
    
    
    
    time_str = '24:00:00'
    main_file = '$HOME/classify_strains/nn/train.py'
    
    params_grid = dict(
    model_type = ['resnet50', 'gru', 'lstm'],
    sample_size_frames_s = [1/25.],
    sample_frequency_s = [15, 90],
    n_epochs = [300],
    is_angle = [True],
    is_CeNDR = [True, False],
    is_reduced = [True, False]
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
        
        dd = base_file.format(time_str=time_str, 
                              cmd_str=cmd_str, 
                              train_file=train_file
                              )
        

#cmd_str = "python {fname} --model_type {model_type} --is_CeNDR --is_angle --sample_size_frames_s 90 --sample_frequency_s 0.1"