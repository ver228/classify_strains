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
    
    
    
    main_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    
    dft_params = OrderedDict(
        is_reduced = True,
        n_batch = 64,
        embedding_size = 256,
        embedding_loss_mixture = 0.1,
        classification_loss_mixture = 1.,
        autoencoder_loss_mixture = 1.
    
    )

    options = [
            (32, 0.1, 1, 1, True),
            (64, 0.1, 1, 1, True),
            (64, 1, 1, 1, True),
            (64, 0.1, 1, 0, False),
            (64, 0.1, 0, 1, False),
            (64, 0.1, 1, 1, False)
            ]

    all_exp = []
    for l, emb, clf, ae, is_r in options:
       args = dft_params.copy()
       args['embedding_size'] = l
       args['embedding_loss_mixture'] = emb
       args['classification_loss_mixture'] = clf
       args['autoencoder_loss_mixture'] = ae
       args['is_reduced'] = is_r
       all_exp.append(args)


    short_add = OrderedDict(
        embedding_size = lambda x : 'L{}'.format(x),
        embedding_loss_mixture = lambda x : 'Emb{}'.format(x),
        classification_loss_mixture = lambda x : 'Clf{}'.format(x) ,
        autoencoder_loss_mixture = lambda x : 'AE{}'.format(x),
        is_reduced = lambda x : 'R' if x else ''
    )
    
    for args in all_exp:
        if args['is_reduced']:
            time_str = '24:00:00'
        else:
            time_str = '48:00:00'
        
        
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
        