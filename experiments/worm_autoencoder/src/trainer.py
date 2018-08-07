#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:50:59 2017

@author: ajaver
"""
from tensorboardX import SummaryWriter

import torch
import os
import shutil
import tqdm
import numpy as np
from collections import OrderedDict


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

class TBLogger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

class TrainerAutoEncoder(object):
    def __init__(self, 
                 model,
                 device,
                 optimizer,
                 criterion,
                 generator,
                 n_epochs,
                 log_dir
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.generator = generator
        self.n_epochs = n_epochs
        self.device = device
        
        # Set the logger
        self.log_dir = log_dir
        self.logger = TBLogger(log_dir)
    
    def fit(self):
        best_loss = 0
        for self.epoch in range(1, self.n_epochs + 1):
            train_metrics = self._train_epoch()
            #save train metrics
            for tag, value in train_metrics.items():
                self.logger.scalar_summary(tag, value, self.epoch)
            
            
            loss = train_metrics['train_loss']
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            state = {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : self.optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, save_dir = self.log_dir)
            
            
    def _train_epoch(self):
        self.model.train()
        pbar = tqdm.tqdm(self.generator)
        
        all_metrics = []
        for input_var in pbar:
            input_var.to(self.device)

            output = self.model(input_var)
            loss = self.criterion(output, input_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            m = self._metrics(output, input_var, loss, is_train=True)
            all_metrics.append(m)
            pbar.set_description(self._pbar_description(m, is_train=False), refresh=False)
            
        return self._metrics_avg(all_metrics)
    
    def _pbar_description(self, m, is_train):
        m_str = ', '.join(['{}: {:}'.format(*x) for x in m])
        d_str = 'Epoch : {} | {}'.format(self.epoch, m_str)
        
        return d_str
        
        
    def _metrics(self, output, target_var, loss, is_train):
            if is_train:
                prefix = 'train_'
             
            tb = [('loss' , loss.item())]
            tb = [(prefix + x, y) for x,y in tb]
            
            return tb
   
    def _metrics_avg(self, m):
        dd = [list(zip(*x)) for x in zip(*m)]
        return OrderedDict([(x[0][0], np.mean(x[1])) for x in dd])
    
