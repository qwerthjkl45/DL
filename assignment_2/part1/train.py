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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
################################################################################


def check_converge(accuracy, accuracy_list):
    converge = False;
    for idx in range(5):
        if abs(accuracy -accuracy_list[-1*(idx+1)]) < 0.001:
            converge = True;
        else:
            return False;
            
    return converge;

def train(config, FIGURE=True):

    assert config.model_type in ('RNN', 'LSTM')
    

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device).cuda();
    else:
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device).cuda();

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate);
    
    #loss_list = [];
    accuracy_list = [];
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        optimizer.zero_grad();
        batch_inputs = batch_inputs.to(device);
        batch_targets = batch_targets.to(device);
        
        out = model(batch_inputs);
        loss_criterion = criterion(out,batch_targets);
        loss_criterion.backward();
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        optimizer.step();
        values, indices = torch.max(out, 1);
        loss = loss_criterion.data[0];
        
        accuracy = ((indices[indices == batch_targets].size())[0])/config.batch_size
        
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            
            accuracy_list += [accuracy];
            if len(accuracy_list) > 5:
                if check_converge(accuracy, accuracy_list):
                    break;

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
            
        
    if not FIGURE:
        plt.figure(1);
        print('Done training.')
        x = np.arange(len(accuracy_list))*10;
        plt.plot(x, accuracy_list, 'r');
        plt.title('Accuracy of VanillaRNN with T={:d}'.format(config.input_length));
        plt.xlabel('Steps');
        plt.ylabel('Accuracy');
        plt.hold(True)
        plt.show();
    
    return accuracy_list;


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    
    #train the model with different input_length:
    DRAW_ACCURACY_WITH_DIFFERENT_T = False;
    if DRAW_ACCURACY_WITH_DIFFERENT_T:
        accuracy_list_with_different_T = [];
        for idx in range(6):
            config.input_length = (idx+1)*5;
            config.train_steps = 1000;
            #if idx > 0:
            #    config.learning_rate = 0.001 *(idx)*3;
            accuracy_list_with_different_T += [train(config)];
            
        x = np.arange(len(accuracy_list_with_different_T[0]))*10;
        fig, ax = plt.subplots();
        color=iter(cm.rainbow(np.linspace(0,1,7)))
        
        for idx in range(6):
            c = next(color);
            learning_rate = 0.001;
            if idx > 0:
                learning_rate = 0.001 *(idx)*3;
            ax.plot(x, accuracy_list_with_different_T[idx], c=c, label = 'T={:d}, lr={:.3f}'.format(((idx+1)*5), learning_rate));
            
        legend = ax.legend(loc='upper center');        
        plt.title('Accuracy of VanillaRNN with different T and steps={:d}'.format(config.train_steps));
        plt.xlabel('Steps');
        plt.ylabel('Accuracy');
        plt.show();
    
        

    # Train the model
    train(config)
