# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, temperature,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size;
        self.seq_length = seq_length;
        self.vocabulary_size = vocabulary_size;
        self.lstm_num_hidden = lstm_num_hidden;
        self.lstm_num_layers = lstm_num_layers;
        self.temperature = temperature;
        self.device = device;
        
        self.lstm = nn.LSTM(self.vocabulary_size, self.lstm_num_hidden, self.lstm_num_layers);
        self.hidden2out = nn.Linear(self.lstm_num_hidden, self.vocabulary_size);
        
        self.h_prev = torch.zeros( self.lstm_num_layers, self.batch_size, self.lstm_num_hidden, device = self.device);
        self.c_prev = torch.zeros( self.lstm_num_layers, self.batch_size, self.lstm_num_hidden, device = self.device);
        
    
    def forward(self, x):
        # x size: # seq_length * batch_size * vacab_size;
        
        output, (h, c) = self.lstm(x, (self.h_prev, self.c_prev));
        hidden2out = self.hidden2out(output)/self.temperature;
        out = nn.functional.softmax(hidden2out, dim=2);
        
        #out = out.view(-1, self.vocabulary_size);
        out = torch.transpose(out, 1, 2);
        
        #out =seq_length * vocab * batch_size
        return out;
