#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:42:24 2017

@author: ajaver
"""
from models import CNNClf, CNNClf1D, Darknet, SimpleDilated, SimpleDilated1D, drn111111
from flow import collate_fn, SkelTrainer
from path import get_path

import tables
import shutil
import os
import tqdm
import datetime
from sklearn.metrics import f1_score
import numpy as np
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

def get_predictions(output, target, topk = (1, 5)):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    _, top_pred = output.topk(maxk)
    top_pred = top_pred.t()
    correct = top_pred.eq(target.view(1, -1).expand_as(top_pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].sum(0)
        correct_k = correct_k.detach().cpu().numpy().tolist()
        res.append(correct_k)
        
    #calculate the global f1 score
    #prefer to use scikit instead of having to program it again in torch
    ytrue = target.detach().cpu().numpy().tolist()
    ypred = top_pred[0].detach().cpu().numpy().tolist()
    
    return (ytrue, ypred, *res)


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)
#%%
def copy_data_to_tmp(fname, copy_tmp):
    if not os.path.exists(copy_tmp):
        os.makedirs(copy_tmp)
            
    bn = os.path.basename(fname)
    new_fname = os.path.join(copy_tmp, bn)
    
    try:
        #make sure the file in tmp is good enough to read an embedding
        with tables.File(new_fname) as fid:
            fid.get_node('/embeddings')[0]
            
    except:
        shutil.copyfile(fname, new_fname)
    return new_fname
        
#%%
class Trainer():
    def __init__(self,
                 cuda_id = 0,
                 n_epochs = 1000,
                batch_size = 4,
                num_workers = 4,
                optimizer = 'adam',
                lr=1e-2,
                weight_decay = 0,
                model_name = 'simple',
                set_type = 'angles',
                is_balance_training = True,
                is_tiny = False,
                is_divergent_set = False,
                root_prefix = None, 
                copy_tmp = None,
                init_model_path = None,
                is_snp = False
                ):
        
        self.is_snp = is_snp
        self.n_epochs = n_epochs
        
        if torch.cuda.is_available():
            print("THIS IS CUDA!!!!")
            dev_str = "cuda:" + str(cuda_id)
        else:
            dev_str = 'cpu'
        
        print(dev_str)
        self.device = torch.device(dev_str)
        
        
        self.fname, self.results_dir_root = get_path(set_type, platform = root_prefix)
        
        if copy_tmp is not None:
            self.fname = copy_data_to_tmp(self.fname, copy_tmp)
            
        
        if self.is_snp:
            return_label = False
            return_snp = True
            criterion = nn.MultiLabelSoftMarginLoss()
        else:
            return_label = True
            return_snp = False
            criterion = nn.CrossEntropyLoss()
        
        self.gen = SkelTrainer(fname = self.fname, 
                              is_balance_training = is_balance_training,
                              is_tiny = is_tiny,
                              is_divergent_set = is_divergent_set,
                              return_label = return_label, 
                              return_snp = return_snp
                          )
        
        self.loader = DataLoader(self.gen, 
                            batch_size = batch_size, 
                            collate_fn = collate_fn,
                            num_workers = num_workers)
        
        self.num_classes = self.gen.num_classes
        
        
        self.embedding_size = self.gen.embedding_size
        if model_name == 'darknet':
            self.model = Darknet([1, 2, 2, 2, 2], self.num_classes)
        elif model_name == 'simple':
            self.model = CNNClf(self.num_classes)
        elif model_name == 'simple1d':
            self.model = CNNClf1D(self.embedding_size, self.num_classes)
        elif model_name == 'drn111111':
            self.model = drn111111(self.num_classes)
        elif model_name == 'simpledilated':
            self.model = SimpleDilated(self.num_classes)
        elif model_name == 'simpledilated1d':
            self.model = SimpleDilated1D(self.embedding_size, self.num_classes)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))
        
        
        if init_model_path:
            assert set_type in init_model_path
            if not os.path.exists(init_model_path):
                init_model_path = os.path.join(self.results_dir_root, init_model_path)
            
            state = torch.load(init_model_path, map_location = dev_str)
            self.model.load_state_dict(state['state_dict'])
            model_name = 'R_' + model_name
        
        self.lr_scheduler = None
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
            #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 10)
        
        else:
            raise ValueError('Invalid optimizer name {}'.format(optimizer))
        
        log_dir_root =  os.path.join(self.results_dir_root, 'logs')
        
        add_postifx = ''
        if is_tiny:
            print("It's me, tiny-log!!!")
            log_dir_root =  os.path.join(self.results_dir_root, 'tiny_log')
            add_postifx = '_tiny'
        elif is_divergent_set:
            print("Divergent set")
            log_dir_root =  os.path.join(self.results_dir_root, 'log_divergent_set')
            add_postifx = '_div'
        
        if self.is_snp:
            log_dir_root = log_dir_root + '_snp'
        
        
        now = datetime.datetime.now()
        bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
        bn += add_postifx
        
        bn = '{}_{}_{}_lr{}_wd{}_batch{}'.format(set_type, bn, optimizer, lr, weight_decay, batch_size)
        print(bn)
    
        self.log_dir = os.path.join(log_dir_root, bn)
        self.logger = SummaryWriter(log_dir = self.log_dir)
        self.criterion = criterion
        
        self.model = self.model.to(self.device)
        
        
        
        
    def _epoch(self, n_iter, epoch, is_train):
        
        if is_train:
            self.model.train()
            self.gen.train()
            log_prefix = 'train_'
        else:
            self.model.eval()
            self.gen.test()
            log_prefix = 'test_'
        
        avg_loss = 0
        all_res = []
        pbar = tqdm.tqdm(self.loader)
        for x_in, y_in in pbar:
            X = x_in.to(self.device)
            target =  y_in.to(self.device)
            del x_in, y_in
    
            pred = self.model(X)
            loss = self.criterion(pred, target)
            
            if is_train:
                self.optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                self.optimizer.step() 
        
            
            desc = log_prefix + 'epoch {} , loss={}'.format(epoch, loss.item())
            pbar.set_description(desc = desc, refresh=False)
    
            #I prefer to add a point at each iteration since hte epochs are very large
            self.logger.add_scalar(log_prefix + 'iter_loss', loss.item(), n_iter)
            n_iter += 1
            avg_loss += loss.item()
            
            if not self.is_snp:
                all_res.append(get_predictions(pred, target))
            
            del loss, pred, X, target

        avg_loss /= len(self.loader)
        tb = [(log_prefix + 'epoch_loss', avg_loss)]
        
        if all_res:
            (ytrue, ypred, pred1, pred5) = map(list, map(itertools.chain.from_iterable, zip(*all_res)))
            f1 = f1_score(ytrue, ypred, average='macro')
            
            tb  += [(log_prefix + 'f1', f1),
                  (log_prefix + 'pred1', np.mean(pred1)*100),
                  (log_prefix + 'pred5', np.mean(pred5)*100)
                  ]
        
        for tt, val in tb:
            self.logger.add_scalar(tt, val, epoch)
        
        return avg_loss, n_iter
    
    
    def train(self):
        best_loss = 1e10
        n_iter_train = 0
        n_iter_test = 0
        for epoch in range(self.n_epochs):
            _, n_iter_train = self._epoch(n_iter_train, epoch, is_train = True)
            with torch.no_grad():
                test_avg_loss, n_iter_test = self._epoch(n_iter_test, epoch, is_train = False)
            
            if self.lr_scheduler:
                self.lr_scheduler.step(test_avg_loss)
            
            is_best = test_avg_loss < best_loss
            best_loss = min(test_avg_loss, best_loss)
            
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : self.optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, save_dir = self.log_dir)#,  filename='checkpoint_{}.pth.tar'.format(epoch))
            
            
#%%

def main(**argkws):
    tt = Trainer(**argkws)
    tt.train()
    
        
if __name__ == '__main__':
    import fire
    fire.Fire(main) 
#    main(
#         n_epochs = 1000,
#        batch_size = 4,
#        num_workers = 1,
#        lr = 1e-2,
#        model_name = 'simple',
#        set_type = 'AE_emb_20180206',#'angles',
#        is_tiny = True,
#        is_divergent_set = False,
#        root_prefix = 'loc')