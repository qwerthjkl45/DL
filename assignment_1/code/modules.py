"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    mean, std = 0, 0.0001;
    self.params['weight'] = np.random.normal(mean, std, in_features* out_features).reshape(out_features, in_features); #d_l+1 * dl
    self.params['bias'] = np.zeros((out_features, 1)); # d_l+1 * 1
    
    self.grads['weight'] = np.zeros((out_features, in_features)); # d_l+1 * 1 #d_l+1 * dl
    self.grads['bias'] = np.zeros((out_features, 1)); # d_l+1 * 1
    
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.input = x;
    out = np.dot(x, self.params['weight'].T) + self.params['bias'].T; #x(n)~ = w(n)*x(n-1) + b(n);
    
    #print('self.output: ', out.shape);
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    #print('#######################')
    #print('LinearModule backward');     
    
    dx = np.dot(dout, self.params['weight']); #((batch_size *D_L) * (D_L* D_(L-1)).T = d_(l-1) * batch_size;
    self.grads['weight'] = np.dot(self.input.T, dout).T #(dl-1* b) * (b * dl )
    self.grads['bias'] = np.sum(dout, axis = 0)
    
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.zeros_like(x); # d_l * batch_size
    out = np.maximum(x, np.zeros(x.shape))

            
    self.input = x;
    self.output = out;
    #print('input: ', x.shape);
    #print('output: ', out.shape);
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout * (self.input > 0)    

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis = 1)
    out = x_exp / np.stack([sum_exp for k in range(x.shape[1])], axis=1)
    
    self.output = out
    self.input_size = x.shape;
    
    
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    x_dx_tilde = np.zeros((self.input_size[0], self.input_size[1], self.output.shape[1])); # batch_size* n* n
    for k in range(self.input_size[0]):
        output = self.output[k];
        output = output.reshape(output.shape[0], 1);
        x_dx_tilde[k] = np.dot(output, output.T);
        x_dx_tilde[k] = -x_dx_tilde[k] + np.diag(self.output[k]);
                
    dx = (np.einsum('ij,ijk->ik',dout, x_dx_tilde)); #   batch_size * D_L
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.log(x) * y # b * c
    out = np.sum(out, axis = 0); # b * 1
    out = -1 * np.mean(out);
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """
   
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # need to check y's dimentsion!!
    dx = -((np.divide(y, x)) + 0); # batch_size* d_l
    dx = dx/x.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
