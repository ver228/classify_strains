#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EmbeddingModel', 'FullLossL2']


class EmbeddingModel(nn.Module):
    def __init__(self, video_model, n_classes, snps_size, embedding_size):
        super(EmbeddingModel, self).__init__()
        self.video_model = video_model
        self.embedding_size = embedding_size
        self.snps_size = snps_size
        
        self.snp_mapper = nn.Sequential(
            nn.Linear(snps_size, 2048), 
            #i do not think a non-linear activation will make a differences here
            nn.Linear(2048, embedding_size)
        )
        self.classification = nn.Linear(embedding_size, n_classes)

    def forward(self, input_d):
        video_input, snps = input_d
        video_embedding = self.video_model(video_input)
        classification = self.classification(video_embedding)
        snps_embedding = self.snp_mapper(snps)
        return classification, video_embedding, snps_embedding


class FullLossL2(nn.Module):
    def __init__(self, embedding_loss_mixture=0.1):
        super().__init__()

        self.embedding_loss_mixture = embedding_loss_mixture
        self.classification_loss = nn.CrossEntropyLoss()
        #self.embedding_loss = nn.MSELoss()

    def forward(self, embedding_output, target_classes):
        classification, video_embedding, snps_embedding = embedding_output
        classification_loss = self.classification_loss(classification,
                                                       target_classes)
        
        # Can't use the Loss layer here because it doesn't like - aej, likely due to autograd gradients
        embedding_loss = F.mse_loss(snps_embedding, video_embedding,
                                    size_average=True)
        loss = classification_loss + \
        self.embedding_loss_mixture * embedding_loss
        return loss