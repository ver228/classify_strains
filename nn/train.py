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


import torch
import tqdm
import tensorflow as tf
import time

from sklearn.metrics import f1_score

from model_resnet import ResNetS, Bottleneck
from model_rnn import RNNModel


from skeletons_flow import SkeletonsFlow, CeNDR_DIVERGENT_SET, SWDB_WILD_ISOLATES

IS_CUDA = torch.cuda.is_available()
if IS_CUDA:
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


#default values
sample_size_frames_s_dflt = 90.
sample_frequency_s_dflt = 1/10

def _h_get_model(num_classes, model_type, n_channels=1):
    
    if model_type == 'resnet50':
        model = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes = num_classes)
    
    elif model_type == 'lstm':
        model = RNNModel('LSTM',
               num_classes = num_classes, 
               input_size = 48, 
               hidden_size = 256, 
               nlayers = 3
               )
    
    elif model_type == 'gru':
        model = RNNModel('GRU',
               num_classes = num_classes, 
               input_size = 48, 
               hidden_size = 256, 
               nlayers = 3
               )
        
    else:
        raise ValueError(model_type)

    return model

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
    
    #calculate the global f1 score
    f1 = f1_score(target.data.numpy(), pred.data[0].numpy(), average='micro')
    return res, f1

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


class Trainer(object):
    def __init__(self, 
                 model,
                 optimizer,
                 criterion,
                 train_generator,
                 test_generator,
                 n_epochs,
                 batch_per_epoch,
                 log_dir,
                 model_type
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_generator = train_generator
        self.test_generator = test_generator
        
        self.n_epochs = n_epochs
        self.log_dir = log_dir
        self.batch_per_epoch = batch_per_epoch
        self.is_rnn = any(x in model_type for x in ['lstm', 'gru'])
        
        # Set the logger
        self.logger = TBLogger(log_dir)
    
    def fit(self):
        best_prec1 = 0
        for self.epoch in range(1, self.n_epochs + 1):
            t_log_data = self._h_epoch(self.model, is_train = True)
            train_loss, train_pred1, train_pred5, train_f1 = map(np.mean, zip(*t_log_data))
            
            v_log_data = self._h_epoch(self.model, is_train = False)
            test_loss, test_pred1, test_pred5, test_f1 = map(np.mean, zip(*v_log_data))
            
            is_best = test_pred1 > best_prec1
            best_prec1 = max(test_pred1, best_prec1)
            
            state = {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : self.optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, save_dir = self.log_dir)
            
            tb_info = {
                    'train_loss':train_loss, 
                    'train_pred1':train_pred1, 
                    'train_pred5':train_pred5, 
                    'train_f1':train_f1,
                    'test_loss':train_loss, 
                    'test_pred1':train_pred1, 
                    'test_pred5':train_pred5,
                    'test_f1':test_f1
                    }
            
            for tag, value in tb_info.items():
                self.logger.scalar_summary(tag, value, self.epoch)
    
    def _h_get_n_batch(self, gen):
        if self.batch_per_epoch is None:
            batch_per_epoch = max(1, len(gen)//gen.n_batch)
        else:
            batch_per_epoch = self.batch_per_epoch
        return batch_per_epoch
    
    def _h_epoch(self,
                 model,
                 is_train,
                 topk = (1, 5)
                 ):
        
        log_data = []
        if is_train:
            gen = self.train_generator
            model.train()
        else:
            gen = self.test_generator
            model.eval()
        
        
        batch_per_epoch = self._h_get_n_batch(gen)
        pbar = tqdm.trange(batch_per_epoch)
        for step in pbar:
            input_var, target_var = self._h_transform_func(next(gen))
            
            output = model(input_var)
            loss = self.criterion(output, target_var)
            if is_train:
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                self.optimizer.zero_grad()
                # Use autograd to compute the backward pass. This call will compute the
                # gradient of loss with respect to all Variables with requires_grad=True.
                # After this call w1.grad and w2.grad will be Variables holding the gradient
                # of the loss with respect to w1 and w2 respectively.
                loss.backward()
                
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
                
                self.optimizer.step()
            
            (prec1, prec5), f1 = accuracy(output, target_var, topk = topk)
            iter_data = (self.epoch, loss.data[0], prec1.data[0], prec5.data[0], f1)
            str_d = 'Epoch: %i [Loss: %.4f, Acc: @1 %.2f | @5 %.2f, F1: %.2f]' % iter_data
            if not is_train:
                str_d = 'Val ' + str_d
            pbar.set_description(str_d)
            
            log_data.append(iter_data[1:])
        
        return log_data
    
    def _h_transform_func(self, D):
        X, Y = D
        
        Y = Y.argmax(axis=1) + 1 # target must be 1, 2, ... number_of_classes
        X = np.rollaxis(X, -1, 1) # the channel dimension must be the second one
        
        Xt = torch.from_numpy(X).float()
        if self.is_rnn:
            Xt = Xt.view(Xt.size(0), -1, Xt.size(-1))
        Yt = torch.from_numpy(Y).long()
        
        if IS_CUDA:
            Xt.cuda()
            Yt.cuda()
            
        input_var = autograd.Variable(Xt)
        target_var = autograd.Variable(Yt)
        
        return input_var, target_var




def main(
        model_type,
        sample_size_frames_s = sample_size_frames_s_dflt,
        sample_frequency_s = sample_frequency_s_dflt,
        n_epochs = 100,
        n_batch_base = 32,
        batch_per_epoch = None,
        is_angle = False,
        is_CeNDR = False,
        is_reduced = False
        ):
    #%%
    if is_CeNDR:
        dataset_str = 'CeNDR'
    else:
        dataset_str = 'SWDB'
    
    
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        data_dir = os.environ['TMPDIR']
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
        data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/'
        data_dir = os.path.join(data_dir,dataset_str)
        
    data_file =  os.path.join(data_dir, dataset_str + '_skel_smoothed.hdf5')
    bn_prefix = dataset_str + '_'
    
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
    
    #%%
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
    
    model = _h_get_model(num_classes, model_type = model_type)
    
    base_name = model_type + '_' + bn_prefix
    base_name = '{}_S{}_F{:.2}'.format(base_name, sample_size_frames_s, sample_frequency_s)
    log_dir = os.path.join(log_dir_root, '%s_%s' % (base_name, time.strftime('%Y%m%d_%H%M%S')))
    
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
    
    t = Trainer(model,
             optimizer,
             criterion,
             train_generator,
             test_generator,
             n_epochs,
             batch_per_epoch,
             log_dir,
             model_type
             )
    t.fit()




if __name__ == '__main__':
  import fire
  fire.Fire(main)    
  
#  main(model_type = 'resnet50',
#       sample_size_frames_s = sample_size_frames_s_dflt,
#        sample_frequency_s = sample_frequency_s_dflt,
#        n_epochs = 2,
#        n_batch_base = 32,
#        batch_per_epoch = 3,
#        is_angle = True,
#        is_CeNDR = True,
#        is_reduced = True
#        )