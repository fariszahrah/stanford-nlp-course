#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    '''
    Highway module which takes a tensor from the convolution layer and outputs a word embedding tensor (by batch)
    '''

    def __init__(self, embed_size):
        '''
        init highway Module 
        @param embed_size(int) : Embedding size (dimensionality)
        '''

        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.proj = nn.Linear(self.embed_size, self.embed_size)
        self.gate = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, conv_out) -> torch.Tensor:
        '''
        feed tensor through highwar
        @param conv_out(Tensor) : Tensor of convolution layer output of size embed_size 
        @returns x_highway(Tensor) : word embedding tensor of size batch_size
        '''
        x_proj = F.relu(self.proj(conv_out))
        x_gate = F.sigmoid(self.gate(conv_out))

        x_highway = (x_proj * x_gate + (1 - x_gate) * conv_out)

        return x_highway
