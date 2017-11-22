#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:55:57 2017

@author: ajaver
"""
from classify.models.resnet import ResNetS, BasicBlock
from classify.models.model_w_embedding import EmbeddingModel

def resnet18_w_embedding(gen, embedding_size):
    video_model = ResNetS(BasicBlock, 
                    [2, 2, 2, 2], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = embedding_size
                    )
    
    
    model = EmbeddingModel(video_model, 
                   gen.n_classes, 
                   gen.n_snps, 
                   embedding_size)
    return model

