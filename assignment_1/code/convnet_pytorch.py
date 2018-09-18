"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__();
    self.classifier = nn.Linear(512, 10);
    
    VGG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'];
    layers = [];
    
    in_channels = 3;
    for layer in VGG:
         out_channels = layer;
         
         if layer == 'M':
             layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)];
         else:
             layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
                        nn.ReLU(inplace=True)]
             in_channels = out_channels;
             
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)];
         
    self.layers = nn.Sequential(*layers);
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.layers(x);
    out = out.view(out.size(0), -1);
    out = self.classifier(out);
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
