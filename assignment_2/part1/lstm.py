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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        
        self.seq_length = seq_length;
        self.input_dim = input_dim;
        self.num_hidden = num_hidden;
        self.num_classes = num_classes;
        self.batch_size = batch_size;
        self.device = device;
        
        # input modulation gate
        self.w_gx = nn.Parameter(torch.randn(num_hidden, input_dim, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.w_gh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.bias_g = nn.Parameter(torch.randn(num_hidden, 1, device=device).normal_(0.0, 0.1), requires_grad=True);
        
        # input gate
        self.w_ix = nn.Parameter(torch.randn(num_hidden, input_dim, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.w_ih = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.bias_i = nn.Parameter(torch.randn(num_hidden, 1, device=device).normal_(0.0, 0.1), requires_grad=True);
        
        # forgoten gate
        self.w_fx = nn.Parameter(torch.randn(num_hidden, input_dim, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.w_fh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.bias_f = nn.Parameter(torch.randn(num_hidden, 1, device=device).normal_(0.0, 0.1), requires_grad=True);
        
        # output gate
        self.w_ox = nn.Parameter(torch.randn(num_hidden, input_dim, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.w_oh = nn.Parameter(torch.randn(num_hidden, num_hidden, device=device ).normal_(0.0, 0.1), requires_grad=True);
        self.bias_o = nn.Parameter(torch.randn(num_hidden, 1, device=device).normal_(0.0, 0.1), requires_grad=True);
        
        self.w_ph = nn.Parameter(torch.randn(num_classes, num_hidden,  device=device).normal_(0.0, 0.1), requires_grad=True);
        self.bias_p = nn.Parameter(torch.randn(num_classes, 1,  device=device).normal_(0.0, 0.1), requires_grad=True);

        

    def forward(self, x):
        
        h_prev = torch.zeros([ self.batch_size, self.num_hidden]).to(self.device);
        c_prev = torch.zeros([ self.batch_size,  self.num_hidden]).to(self.device);
        for idx in range(self.seq_length):
            tmp_x = x[:, idx].view(-1, 1); # batch_size * 1
            g = torch.tanh(tmp_x.mm(torch.transpose(self.w_gx, 0, 1))+ h_prev.mm(torch.transpose(self.w_gh, 0, 1)) +torch.transpose(self.bias_g, 0, 1)); # batch_size * num_hidden 
            i = torch.sigmoid(tmp_x.mm(torch.transpose(self.w_ix, 0, 1))+ h_prev.mm(torch.transpose(self.w_ih, 0, 1)) +torch.transpose(self.bias_i, 0, 1)); # batch_size * num_hidden 
            f = torch.sigmoid(tmp_x.mm(torch.transpose(self.w_fx, 0, 1))+ h_prev.mm(torch.transpose(self.w_fh, 0, 1)) +torch.transpose(self.bias_f, 0, 1)); # batch_size * num_hidden 
            o = torch.sigmoid(tmp_x.mm(torch.transpose(self.w_ox, 0, 1))+ h_prev.mm(torch.transpose(self.w_oh, 0, 1)) +torch.transpose(self.bias_o, 0, 1)); # batch_size * num_hidden 
            
            c = (g*i) + (c_prev*f);
            c_prev = c;
            
            h = (torch.tanh(c)) * o;
            h_prev = h;
            
        p = h.mm(torch.transpose(self.w_ph, 0, 1)) + torch.transpose(self.bias_p, 0, 1); # batch_size * num_classes
        out = nn.functional.softmax(p);
        return out;
        
