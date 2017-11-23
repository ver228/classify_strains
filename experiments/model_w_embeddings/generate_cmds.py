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

#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

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
    #add the parent directory to the log results
    pdir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
    save_dir = os.path.join(os.pardir, 'cmd_scripts', pdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    time_str = '24:00:00'
    
    main_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    
    dft_params = OrderedDict(
        model_name = 'resnet18_w_embedding',
        is_reduced = True,
        embedding_size = 256,
        sample_size_seconds = 10,
        sample_frequency_s = 0.04,
        n_batch = 32,
        n_epochs = 200,
        embedding_loss_mixture = None,
        loss_type = None
    )

    all_exp = []
    options = [
            ('l2', 0.),
            ('l2', 0.01),
            ('l2', 0.001),
            ('l1', 0.01),
            ('l1', 0.001),
            ]

    for lt, elm in options:
        args = dft_params.copy()
        args['loss_type'] = lt
        args['embedding_loss_mixture'] = elm
        all_exp.append(args)


    short_add = OrderedDict(
        loss_type = lambda x : x if x else x,
        embedding_loss_mixture = lambda x : '{}'.format(x) if x else x,
        model_name = lambda x : x
    )
    
    for args in all_exp:
        args_d = ' '.join(['--{} {}'.format(*d) for d in args.items()])
        cmd_str = 'python {} {}'.format(main_file, args_d)
        
        train_file = 'CeNDR_skel_smoothed.hdf5'

        f_content = base_file.format(time_str=time_str, 
                              cmd_str=cmd_str, 
                              train_file=train_file
                              )

        f_name = [func(args[k]) for k,func in short_add.items()]
        f_name = '_'.join([x for x in f_name if x]) + '.sh'
        f_name = os.path.join(save_dir, f_name)
        
        with open(f_name, 'w') as fid:
            fid.write(f_content)
        