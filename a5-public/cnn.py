#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    '''
    word based convolution Module
    '''
    def __init__(self, embed_size, out_channels, word_len, padding=1 ,kernel_size=5):
        '''
        convolution layer init
        @param embed_size(int) : size of character embeddings

        '''
        super(CNN, self).__init__()
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(embed_size, out_channels, kernel_size, padding=padding) 
        self.pool = nn.MaxPool1d(word_len - kernel_size + 1 + (2*padding))


    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        '''
        pass character embeddings through Conv1d layer,relu, and maxpool
        @param x_reshaped(Tensor): tensor of character leverl embeddings
        @returns x_conv (Tensor): tensor of word embedding of size 
        '''
        print('Embedding size: ', self.embed_size)
        print('X_reshaped size: ',x_reshaped.size())
        x = self.conv1(x_reshaped)
        x = F.relu_(x)
        print('x_size after conv with out_channels: ',self.out_channels, ',  ',x.size())
        #x = nn.ReLU(x)
        x_conv = self.pool(x).squeeze()
        print('x_conv_size: ', x_conv.size())
        
        return x_conv

