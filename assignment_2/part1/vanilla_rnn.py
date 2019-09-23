################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cuda:0'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        
        self.seq_length = seq_length;
        self.input_dim = input_dim;
        self.num_hidden = num_hidden;
        self.num_classes = num_classes;
        self.batch_size = batch_size;
        self.device = device;
        
        #self.i2h = nn.Linear(input_dim, num_hidden, bias=True)
        #self.h2h = nn.Linear(num_hidden, num_hidden, bias=True)
        self.w_i2h = nn.Parameter(torch.randn(num_hidden, input_dim, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.w_h2h = nn.Parameter(torch.randn(num_hidden, num_hidden,  device=device).normal_(0.0, 0.1), requires_grad=True);
        self.w_h2p = nn.Parameter(torch.randn(num_classes, num_hidden,  device=device).normal_(0.0, 0.1), requires_grad=True);
        self.bias_h = nn.Parameter(torch.randn(num_hidden, 1, device=device).normal_(0.0, 0.1), requires_grad=True);
        self.bias_p = nn.Parameter(torch.randn(num_classes, 1,  device=device).normal_(0.0, 0.1), requires_grad=True);

    def forward(self, x):
        # Implementation here ...
        #h_prev = torch.zeros([self.num_hidden, 1]).to(self.device);
        h_prev = torch.zeros([ self.batch_size, self.num_hidden]).to(self.device);
        for idx in range(self.seq_length):
            tmp_x = x[:, idx].view(-1, 1); # batch_size * 1
            h = torch.tanh(tmp_x.mm(torch.transpose(self.w_i2h, 0, 1))+ h_prev.mm(torch.transpose(self.w_h2h, 0, 1)) +torch.transpose(self.bias_h, 0, 1)); # batch_size * num_hidden
            h_prev = h;
            
        p = h.mm(torch.transpose(self.w_h2p, 0, 1)) + torch.transpose(self.bias_p, 0, 1); # batch_size * num_classes
        out = nn.functional.softmax(p);
        return out;
