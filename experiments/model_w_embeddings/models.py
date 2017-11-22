#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:55:57 2017

@author: ajaver
"""
from classify.models.resnet import ResNetS, BasicBlock
from classify.models.model_w_embeddings import EmbeddingModel

def resnet18_w_embeddings(gen, ):
    model_video = ResNetS(BasicBlock, 
                    [2, 2, 2, 2], 
                    avg_pool_kernel = (7,1),
                    n_channels = gen.n_channels, 
                    num_classes = gen.n_classes
                    )
    EmbeddingModel(model_video, )
    return model

