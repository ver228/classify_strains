#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:41 2017

@author: ajaver
"""
import os
from collections import OrderedDict

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
cp $WORK/classify_strains/train_set/{mask_file} $TMPDIR/
cp $WORK/classify_strains/train_set/{feat_file} $TMPDIR/

{cmd_str}
## This tells the batch manager to execute the program cudaexecutable in the cuda directory of the users home directory.
'''

if __name__ == '__main__':
    #add the parent directory to the log results
    pdir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
    save_dir = os.path.join(os.pardir, 'cmd_scripts', pdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    time_str = '48:00:00'
    
    main_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    
    dft_params = OrderedDict(
        model_name='AE3D', 
         batch_size = 3,
         snippet_size = 255,
         roi_size = 128,
         n_epochs = 1000,
         embedding_size = 256,
         max_n_frames = -1,
         pretrained_path = None,
         emb_reg_loss_mix = 0
    )
    all_exp = []
    
    for em_s in [64, 256]:
        for mod in ['AE3D', 'AE2D', 'AE2D_RNN']:
            if 'AE2D' in mod:
                i_dd = [0, 0.1]
            else:
                i_dd = [0]
                
            for e_reg in i_dd:
                args = dft_params.copy()
                args['model_name'] = mod
                args['embedding_size'] = em_s
                args['emb_reg_loss_mix'] = e_reg
                all_exp.append(args)
            
    
#    pretrained = ['/work/ajaver/classify_strains/results/snippet_autoencoder/AE3D_L256_20171201_092555/checkpoint.pth.tar']
#    for fname in pretrained:
#        args = dft_params.copy()
#        args['pretrained_path'] = fname
#        all_exp.append(args)
    

    short_add = OrderedDict(
         model_name = lambda x : x,
         embedding_size = lambda x : 'E{}'.format(x),
         emb_reg_loss_mix = lambda x : '' if x <= 0 else 'mix{}'.format(x)
     )
    
    for args in all_exp:
        args_d = ' '.join(['--{} {}'.format(*d) for d in args.items()])
        cmd_str = 'python {} {}'.format(main_file, args_d)
        
        mask_file = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021.hdf5'
        feat_file = 'BRC20067_worms10_food1-10_Set10_Pos5_Ch6_16052017_165021_featuresN.hdf5'

        f_content = base_file.format(time_str = time_str, 
                              cmd_str = cmd_str, 
                              mask_file = mask_file,
                              feat_file = feat_file
                              )

        f_name = [func(args[k]) for k,func in short_add.items()]
        f_name = '_'.join([x for x in f_name if x]) + '.sh'
        f_name = os.path.join(save_dir, f_name)
        
        with open(f_name, 'w') as fid:
            fid.write(f_content)
        