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
    
    mean = 0.0
    std = 0.0001

    '''
    self.params['weight'] = numpy.random.normal(mean, std, (in_features, out_features))
    self.params['bias'] = np.zeros((in_features, out_features), dtype=np.float)

    self.grads['weight'] = np.zeros((in_features, out_features), dtype=np.float)
    self.grads['bias'] = np.zeros((in_features, out_features), dtype=np.float)
    '''

    self.params['weight'] = np.random.normal(mean, std, (out_features, in_features))
    self.params['bias'] = np.zeros((out_features, 1), dtype=np.float)

    self.grads['weight'] = np.zeros((out_features, in_features), dtype=np.float)
    self.grads['bias'] = np.zeros((out_features, 1), dtype=np.float)


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
    print("-----------------Linear F----------------")
    
    out = self.params['weight'].dot(x) + self.params['bias']

    self.x = x
    self.out = out
    
    print("w:", self.params['weight'].shape)
    print("x:", x.shape)
    print("b:", self.params['bias'].shape)
    print("out:", out.shape)

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
    
    print("-----------------Linear B----------------")

    dxbar_dw = np.ndarray(shape=(self.out.shape[0], self.params['weight'].shape[0], self.params['weight'].shape[1]))
    i_idx = range(dxbar_dw.shape[0])

    for idx, value in enumerate(self.x):        
        dxbar_dw[i_idx, i_idx, idx] = value

    dxbar_db = np.identity(self.out.shape[0])

    dxbar_dx = self.params['weight']

    self.grads['weights'] = dout.dot(dxbar_dw).T / self.x.shape[1]
    self.grads['bias'] = dout.dot(dxbar_db).T / self.x.shape[1]

    dx = dout.dot(dxbar_dx)
 
    print("dout:", dout.shape)    
    print("dxbar_dw:", dxbar_dw.shape)
    print("dxbar_db:", dxbar_db.shape)
    print("dxbar_dx:", dxbar_dx.shape)
    print("dx", dx.shape)

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
    
    print("-----------------ReLU F----------------")

    out = np.maximum(0, x)
    
    self.x = x
    self.out = out
    
    print("x", x.shape)
    print("out:", out.shape)


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
 
    print("-----------------ReLU B----------------")

    dx_dxbar = np.zeros(shape=(self.x.shape[0], self.x.shape[0]))
    gtz_idx = np.argwhere(self.x > 0.0)
    dx_dxbar[gtz_idx,gtz_idx] = 1
    
    dx = dout.dot(dx_dxbar)

    print("dout:", dout.shape)
    print("dx_dxbar:", dx_dxbar.shape)
    print("dx", dx.shape)


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
    
    print("-----------------Softmax F----------------")

    x_exp = np.exp(x)    
    sum_exp = np.sum(x_exp, axis=0).reshape(1, x.shape[1])

    out = x_exp / sum_exp
    
    self.x = x
    self.out = out
    self.x_exp = x_exp
    self.sum_exp = sum_exp

    print("x_exp:", x_exp.shape)
    print("x:", x.shape)
    print("sum_exp:", sum_exp.shape)
    print("out:", out.shape)

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
    
    print("-----------------Softmax B----------------")
    
    dout = dout.T

    print("dout:", dout.shape)
    print("x_exp:", self.x_exp.shape)
    print("sum_exp:", self.sum_exp.shape)
    print("self.x_exp * self.sum_exp:", (self.x_exp * self.sum_exp).shape)

    dx_dxbar = (np.diag(self.x_exp * self.sum_exp) - self.x_exp.dot(self.x_exp.T)) / (self.sum_exp) ** 2
     
    dx = dout.dot(dx_dxbar)

    print("dout:", dout.shape)
    print("dx_dxbar:", dx_dxbar.shape)
    print("dx", dx.shape)


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
 
    print("-----------------Cross F----------------")

    x = x.T
    out = -np.log(x[:,np.argmax(y,axis = 1)])
    
    print("x:", x.shape)
    print("y:", y.shape)
    print("argmax: ", np.argmax(y,axis = 1).shape)
    print("out:", out.shape)


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
    
    print("-----------------Cross B----------------")

    dx = - y.T / x
    
    print("x:", x.shape)
    print("y:", y.shape)
    print("dx", dx.shape)


    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
