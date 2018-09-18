"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  
  pred_class = np.argmax(predictions, axis = 1);
  t_class = np.argmax(targets, axis = 1);
  accuracy = np.sum(pred_class == t_class) /predictions.shape[0];
  
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  if FLAGS.batch_size:
    batch_size = int(FLAGS.batch_size);
    
  if FLAGS.learning_rate:
    learning_rate = float(FLAGS.learning_rate);
    
  if FLAGS.max_steps:
    max_steps = int(FLAGS.max_steps);
  
  if FLAGS.max_steps:
    eval_freq = int(FLAGS.eval_freq);
      

  cifar10 = cifar10_utils.get_cifar10();
  mlp = MLP(3* 32* 32, dnn_hidden_units, 10, learning_rate);
  cifar10_train = cifar10['train'];
  
  # get all test image labels and features:
  cifar10_test = cifar10['test']; 
  x_test, y_target = cifar10_test.images, cifar10_test.labels;
  x_test = x_test.reshape((x_test.shape[0], -1));
  
  
  steps = 0;
  accuracy_rates = [];  
  tmp_accuracy_rates = [];
  loss_lists = []; 
  tmp_loss_lists = []; 
  
  while (steps <= max_steps):
      epoch_completed = cifar10_train.epochs_completed;
      x, y = cifar10_train.next_batch(batch_size);
      #x_test, y_target = cifar10_test.next_batch(batch_size);
      x = x.reshape((batch_size, -1));
      
      
      out = mlp.forward(x);
      
      crossentropy_loss = mlp.loss_forward(out, y);
      dout = mlp.loss_backward(out, y);
      
      mlp.backward(dout);
      steps = steps + 1;
      
      if ((steps % eval_freq) == 0):
          y_test = mlp.forward(x_test)
          rate = accuracy(y_test, y_target)
          print('---accuracy: ', rate, '---');
          tmp_accuracy_rates += [rate]
          tmp_loss_lists += [crossentropy_loss];
      
      if not (cifar10_train.epochs_completed == epoch_completed):
           print('===finish one epoch===');
           print("Average accuracy: ", sum(tmp_accuracy_rates)/len(tmp_accuracy_rates));
           accuracy_rates += [sum(tmp_accuracy_rates)/len(tmp_accuracy_rates)];
           loss_lists += [sum(tmp_loss_lists)/len(tmp_loss_lists)];
           tmp_accuracy_rates = [];
           tmp_loss_lists = [];
           
  print('finish!');
           
  t = np.arange(1, cifar10_train.epochs_completed + 1, 1);
  a = np.asarray(accuracy_rates);
  l = np.asarray(loss_lists);
  
  plt.figure(1);
  plt.xticks(t);
  plt.ylabel('accuracy');
  plt.xlabel('epoch');
  plt.plot(t, a, 'b');
  plt.show();
  
  plt.figure(2);
  plt.xticks(t);
  plt.ylabel('loss');
  plt.xlabel('epoch');
  plt.plot(t, l, 'b');
  plt.show();
  
  print(a);
  print(l);
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
