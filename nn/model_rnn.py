#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:35:24 2017

modified from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, num_classes, input_size, hidden_size, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn = getattr(nn, rnn_type)(input_size, 
                                          hidden_size, 
                                          nlayers, 
                                          dropout=dropout,
                                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.num_classes = num_classes

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data #this is only to get the data type of the hidden layers
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_())
    
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
            
        x, _ = self.rnn(x, hidden)  
        x = self.drop(x)
        
        # Decode hidden state of last time step
        out  = self.fc(x[:, -1, :])
        return out
        
if __name__ == '__main__':
    mod = RNNModel(rnn_type = 'GRU',
                   num_classes = 100, 
                   input_size = 48, 
                   hidden_size = 128, 
                   nlayers = 3
                   )