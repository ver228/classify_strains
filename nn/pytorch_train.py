#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:50:59 2017

@author: ajaver
"""
import torch
from torch import nn, autograd
import sys
import os
import math
import numpy as np
import shutil

import tqdm
import tensorflow as tf
import time

from pytorch_models import ResNetS, Bottleneck
from skeletons_flow import SkeletonsFlow, CeNDR_DIVERGENT_SET, SWDB_WILD_ISOLATES

IS_CUDA = torch.cuda.is_available()
if IS_CUDA:
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
      
def _h_epoch(model, gen, is_train, batch_per_epoch = None, topk = (1, 5)):
    
    log_data = []
    if is_train:
        model.train()
    else:
        model.eval()
    
    if batch_per_epoch is None:
        batch_per_epoch = max(1, len(gen)//gen.n_batch)
    
    pbar = tqdm.trange(batch_per_epoch)
    for step in pbar:
        X,Y = next(gen)
        Y = Y.argmax(axis=1) + 1 # target must be 1, 2, ... number_of_classes
        X = np.rollaxis(X, -1, 1) # the channel dimension must be the second one
        Xt = torch.from_numpy(X).float()
        Yt = torch.from_numpy(Y).long()
        if IS_CUDA:
            Xt.cuda()
            Yt.cuda()
        
        input_var = autograd.Variable(Xt)
        target_var = autograd.Variable(Yt)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output, target_var, topk = topk)
        
        if is_train:
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()
            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Variables with requires_grad=True.
            # After this call w1.grad and w2.grad will be Variables holding the gradient
            # of the loss with respect to w1 and w2 respectively.
            loss.backward()
            optimizer.step()
        
        iter_data = (epoch, loss.data[0], prec1.data[0], prec5.data[0])
        str_d = 'Epoch: %i [Loss: %.4f, Acc: @1 %.2f | @5 %.2f]' % iter_data
        if not is_train:
            str_d = 'Val ' + str_d
        pbar.set_description(str_d)
        
        log_data.append(iter_data[1:])
    
    return log_data



if __name__ == '__main__':
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        data_dir = os.environ['TMPDIR']
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
        data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/'

    
    sample_size_frames_s_dflt = 90.
    sample_frequency_s_dflt = 1/10
    
    sample_size_frames_s = sample_size_frames_s_dflt
    sample_frequency_s = 1/10
    
    n_epochs = 1
    n_batch_base = 32
    batch_per_epoch = None
    
    is_angle = True
    is_CeNDR = True
    is_reduced = True
    
    #%%
    
    if is_CeNDR:
        data_file =  os.path.join(data_dir, 'CeNDR', 'CeNDR_skel_smoothed.hdf5')
        bn_prefix = 'CeNDR_'
    else:
        data_file = os.path.join(data_dir, 'SWDB', 'SWDB_skel_smoothed.hdf5')
        bn_prefix = 'SWDB_'
    
    valid_strains = None
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
                           n_batch = n_batch, 
                           set_type = 'train',
                           valid_strains = valid_strains,
                           is_angle = is_angle
                           )
    test_generator = SkeletonsFlow(data_file = data_file, 
                           n_batch = n_batch, 
                           set_type = 'test',
                           valid_strains = valid_strains,
                           is_angle = is_angle
                           )
    
    
    
    
    
    X,Y = next(train_generator)
    num_classes = Y.shape[1]
    #%%
    model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes = num_classes)
    
    base_name = bn_prefix + type(model).__name__
    base_name = 'S{}_F{:.2}_{}'.format(sample_size_frames_s, sample_frequency_s, base_name)
    log_dir = os.path.join(log_dir_root, 'logsN', '%s_%s' % (base_name, time.strftime('%Y%m%d_%H%M%S')))
    
    print(model)  
    print(list(train_generator.skeletons_ranges['strain'].unique()))
    print(train_generator.n_batch)
    
    #%%
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if IS_CUDA:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model.cuda()
        criterion.cuda()
        model = nn.parallel.DistributedDataParallel(model)
    
    # Set the logger
    logger = TBLogger(log_dir)
    
    for epoch in range(1, n_epochs + 1):
        t_log_data = _h_epoch(model, 
                              train_generator,  
                              batch_per_epoch = batch_per_epoch, 
                              is_train = True)
        train_loss, train_pred1, train_pred5 = map(np.mean, zip(*t_log_data))
        
        v_log_data = _h_epoch(model, 
                              test_generator,  
                              batch_per_epoch = batch_per_epoch, 
                              is_train = False)
        test_loss, test_pred1, test_pred5 = map(np.mean, zip(*v_log_data))
        
        