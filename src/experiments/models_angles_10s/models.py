#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:55:57 2017

@author: ajaver
"""
from classify.models.resnet import ResNetS, BasicBlock, Bottleneck

def resnet18(gen):
    model = ResNetS(BasicBlock, 
                    [2, 2, 2, 2], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = gen.n_classes
                    )
    return model

def resnet34(gen):
    model = ResNetS(BasicBlock, 
                    [3, 4, 6, 3], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = gen.n_classes
                    )
    return model

def resnet50(gen):
    model = ResNetS(Bottleneck, 
                    [3, 4, 6, 3], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = gen.n_classes
                    )
    return model


def resnet101(gen):
    model = ResNetS(Bottleneck, 
                    [3, 4, 23, 3], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = gen.n_classes
                    )
    return model
