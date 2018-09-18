"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  predictions = predictions.data.numpy();
  
  pred_class = np.argmax(predictions, axis = 1);
  t_class = np.argmax(targets, axis = 1);
  accuracy = np.sum(pred_class == t_class) /predictions.shape[0];
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  if FLAGS.batch_size:
    batch_size = int(FLAGS.batch_size)

  cifar10 = cifar10_utils.get_cifar10();
  convNet = ConvNet(3, 10);
  print(convNet);
  lossfunc = nn.CrossEntropyLoss();
  optimizer = torch.optim.Adam(convNet.parameters(),lr=LEARNING_RATE_DEFAULT)
  
  cifar10_train = cifar10['train'];  
  # get all test image labels and features:
  cifar10_test = cifar10['test']; 
  
  while (cifar10_train.epochs_completed < 10):
      x, y = cifar10_train.next_batch(batch_size);
      x_test, y_target = cifar10_test.next_batch(batch_size);
      x = torch.autograd.Variable(torch.from_numpy(x));
      y = torch.autograd.Variable(torch.from_numpy(y).long());
      x_test = torch.autograd.Variable(torch.from_numpy(x_test));
      
      #x_test = x_test.reshape((batch_size, -1));
      optimizer.zero_grad();
      out = convNet(x);
      
      loss = lossfunc(out,torch.max(y, 1)[1]);
      loss.backward();
      optimizer.step();
      
      y_test = convNet(x_test);
      rate = accuracy(y_test, y_target)
      
      #print('loss: ', crossentropy_loss);
      print("Accuracy:", rate)
      
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
