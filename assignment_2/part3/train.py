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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
import matplotlib.pyplot as plt

################################################################################

def generate_new_stings(model, input_chrarcter, vocab_size, seq_length):
    
    output_string =  (torch.Tensor(seq_length, vocab_size)).type(torch.LongTensor).to("cuda:0");
    prev_string = input_chrarcter;
    
    for idx in range(seq_length):
        batch_inputs_onehot = one_hot(prev_string, vocab_size); # seq_length * batch_size * vacab_size;
        out = model(batch_inputs_onehot); # seq_length * vocab * batch_size
        #out = torch.transpose(out, 1, 2); # seq_length * batch_size * vacab_size;
        if idx < seq_length - 1:
            values, indices = torch.max(out, 1);
            prev_string[idx + 1, :] = indices[idx, :];
            
        output_string[idx, :] = out[idx , :, 0]; #seq_length * vacab_size
    val, ind = torch.max(output_string, 1);
    print(ind);    
    return ind;

def one_hot(x, vocab_size):

    x_ = torch.unsqueeze(x, 2);
    out = torch.FloatTensor(x.size()[0], x.size()[1], vocab_size).to("cuda:0");
    out.zero_();
    out.scatter_(2, x_, 1);
    return out

def train(config, CHOICES):
    
    # Initialize the device which to run the model on
    #device = torch.device(config.device)# fix this!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize the model that we are going to use

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length );  # fixme
    model = TextGenerationModel( config.batch_size, config.seq_length, dataset.vocab_size, config.temperature).cuda();
    if (CHOICES['LOAD_BEST_MODEL']):
        model.load_state_dict(torch.load('./model_parameter.txt'));
    #print(model.state_dict());
    
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate);
    if (CHOICES['LOAD_BEST_MODEL']):
        optimizer.load_state_dict(torch.load('./model_optimizer.txt'));
    accuracy_list = [];
    loss_list = [];
    string_list = [];
    tmp_accuracy = 0;
    
    a = 76;
    while (tmp_accuracy == 0) or (accuracy_list[-1] >0.85): 
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()
            
            batch_inputs = torch.stack(batch_inputs)[:,:, None].view(config.seq_length, -1).to(device); # sequ_length * batch_size
            batch_targets = torch.stack(batch_targets)[:,:, None].view(config.seq_length, -1).to(device); # sequ_length * batch_size
            
            if not((int(batch_inputs.size()[1])) == config.batch_size):
                continue;
                
            #print(dataset.convert_to_string(batch_inputs[:, 0].cpu().numpy())); 
            
            batch_inputs_onehot = one_hot(batch_inputs, dataset.vocab_size); # seq_length * batch_size * vacab_size;
            optimizer.zero_grad();
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm);
            out = model(batch_inputs_onehot);
            
            values, indices = torch.max(out, 1);
            
            loss_criterion = criterion(out,batch_targets);
            loss_criterion.backward();
            optimizer.step();
            
            loss = loss_criterion.data[0]/(config.seq_length);
            values, indices = torch.max(out, 1);
            
            accuracy = ((indices[indices == batch_targets].size())[0])/(config.batch_size*config.seq_length);

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            if step % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                          "Accuracy = {:.2f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                            int(config.train_steps), config.batch_size, examples_per_second,
                            accuracy, loss))
                            
                # generate sentences
                if step % 50000 == 0 and CHOICES['GENERATE_FIVE_SENTENCES']:                            
                    model.eval();                    
                    test_input = (torch.Tensor(batch_inputs.size())).type(torch.LongTensor).to(device);
                    a = a + 1;
                    test_input = test_input.fill_(a);
                    output_string = generate_new_stings(model, test_input, dataset.vocab_size, config.seq_length);  
                    tmp = dataset.convert_to_string(output_string.cpu().numpy().tolist());
                    string_list += [tmp];
                    print(tmp);
                    print('---')     
                    
                    model.train();
                # save parameter
                torch.save(model.state_dict(), './model_parameter{:d}.txt'.format(step));
                torch.save(optimizer.state_dict(), './model_optimizer{:d}.txt'.format(step));                    
                
                
                if (CHOICES['DRAW_ACCURACY_PLOT']):
                    accuracy_list += [accuracy];  
                    loss_list += [loss]; 
                

            if step == config.sample_every:
                # Generate some sentences by sampling from the model
                pass
            
            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
                
            if (CHOICES['GENERATE_FIVE_SENTENCES']) and (len(string_list) == 5):
                break;
        
        if (CHOICES['GENERATE_FIVE_SENTENCES']) and (len(string_list) == 5):
                break;
                        
        
        print("============ finish {} epoch ============ ".format(len(accuracy_list)));
        
    torch.save(model.state_dict(), './model_parameter.txt');
    torch.save(optimizer.state_dict(), './model_optimizer.txt');
    print('Done training.');
    
    if (CHOICES['GENERATE_FIVE_SENTENCES']):
    
        if (CHOICES['DRAW_ACCURACY_PLOT']):
            fig, ax = plt.subplots();
            ax.plot(np.arange(len(accuracy_list)), accuracy_list, 'r', label = 'accuracy');
            ax.plot(np.arange(len(accuracy_list)), loss_list, 'b', label = 'loss');
            legend = ax.legend(loc='upper center');      
            plt.xlabel('Steps');
            plt.title('loss and accuracy of LSTM in 2000 steps');
            plt.show();
        
        for idx in range(5):
            print('====')
            print(string_list[idx]);
    

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--temperature', type=float, default=1, help='temperature')

    config = parser.parse_args()
    torch.backends.cudnn.enabled=False
    
    CHOICES = {
        'LOAD_BEST_MODEL': True,
        'GENERATE_FIVE_SENTENCES': False,
        'TRY_WITH_DIFFERENT_TEMPERATURE': False,
        'DRAW_ACCURACY_PLOT': False
    }
    
    if CHOICES['TRY_WITH_DIFFERENT_TEMPERATURE']:
        ts = [0.5, 1, 2]
        for _, t in enumerate(ts):
            print('========= t =', t, '=========');
            config.temperature = t;
            train(config, CHOICES);
    
    
    
    # Train the model
    train(config, CHOICES)
