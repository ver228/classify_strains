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
    #add the parent directory to the log results
    pdir = os.path.split(os.path.dirname(__file__))[-1]
    save_dir = os.path.join(os.pardir, 'cmd_scripts', pdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_str = '24:00:00'
    main_file = '$HOME/classify_strains/nn/train.py'
    
    dft_params = OrderedDict(
        model_name = None,
        dataset = 'SWDB',
        data_file = None, #get defaults
        is_reduced = True,
        sample_size_seconds = 10,
        sample_frequency_s = 0.04,
        n_batch = 32,
        transform_type = None,
        is_normalized = None,   
        n_epochs = 200
    )

    all_exp = []
    #test1 -> resnet18 on several data transforms with and without normalization
    transform_options = ['xy', 'angles', 'eigenworms', 'eigenworms_full']
    normalize_options = [False, True]

    for tp, np in product(transform_options, normalize_options):
        args = dft_params.copy()
        args['model_name'] = 'resnet18'
        args['transform_type'] = tp
        args['is_normalized'] = np
        all_exp.append(args)

    #test12-> different resnets on unnormalized angles
    model_options = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    for mod in model_options:
        args = dft_params.copy()
        args['model_name'] = mod
        args['transform_type'] = 'angles'
        args['is_normalized'] = False
        all_exp.append(args)

    short_add = OrderedDict(
        model_name = lambda x : x,
        transform_type = lambda x : x,
        is_normalized = lambda x : 'N' if x else ''
    )
    
    for args in all_exp:
        args_d = ' '.join(['--{} {}'.format(*d) for d in args.items()])
        cmd_str = 'python {} {}'.format(main_file, args_d)
        
        train_file = '{}_skel_smoothed.hdf5'.format(args['dataset'])

        f_content = base_file.format(time_str=time_str, 
                              cmd_str=cmd_str, 
                              train_file=train_file
                              )

        f_name = [func(args[k]) for k,func in short_add.items()]
        f_name = '_'.join([x for x in f_name if x]) + '.sh'
        f_name = os.path.join(save_dir, f_name)
        
        with open(f_name, 'w') as fid:
            fid.write(f_content)
        