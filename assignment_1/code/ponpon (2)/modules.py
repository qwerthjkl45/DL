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
    self.params['bias'] = np.zeros((out_features, ), dtype=np.float)

    self.grads['weight'] = np.zeros((out_features, in_features), dtype=np.float)
    self.grads['bias'] = np.zeros((out_features, ), dtype=np.float)


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
    x = x.T

    out = (self.params['weight'].dot(x).T + self.params['bias'])

    self.x = x.T
    self.out = out.T
    
    #print("w:", self.params['weight'].shape)
    #print("x:", x.shape)
    #print("b:", self.params['bias'].shape)
    #print("out:", out.shape)

    #print("I:", x)
    #print("O:", out)
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

    dout = dout
    self.out = self.out
    self.x = self.x

    #print("dout:", dout.shape)    

    #dxbar_dw = np.ndarray(shape=(self.out.shape[0], self.params['weight'].shape[0], self.params['weight'].shape[1]))
    #i_idx = range(dxbar_dw.shape[0])

    #for idx, value in enumerate(self.x):        
    #    dxbar_dw[i_idx, i_idx, idx] = value
    
    dxbar_dw = []
    dxbar_db = []
    dxbar_dx = []

    #print("x", self.x.shape)
    #print("out", self.out.shape)
        

    '''
    for x_idx in range(self.x.shape[0]):
        dxbar_dw_one = np.zeros(shape=(self.out.shape[0], self.params['weight'].shape[0], self.params['weight'].shape[1]))
        for i in range(self.out.shape[0]):
            dxbar_dw_one[i,i,:] = self.x[x_idx,:]

        dxbar_dw += [dxbar_dw_one]
        dxbar_db += [np.identity(self.out.shape[0])]
        dxbar_dx += [self.params['weight']]


    dxbar_dw = np.array(dxbar_dw)
    dxbar_db = np.array(dxbar_db)
    dxbar_dx = np.array(dxbar_dx)
    '''
    #print("dxbar_dw:", dxbar_dw.shape)
    #print("dxbar_db:", dxbar_db.shape)
    #print("dxbar_dx:", dxbar_dx.shape)
    '''
    self.grads['weight'] = np.mean(np.einsum('ij,ijkl->ikl', dout, dxbar_dw), axis=0) * self.x.shape[0]
    self.grads['bias'] = np.mean(np.einsum('ij,ijk->ik', dout, dxbar_db), axis=0)
    self.grads['bias'] = self.grads['bias'].reshape(self.grads['bias'].shape[0], 1) * self.x.shape[0]
    #self.grads['weight'] = np.squeeze(dout.dot(dxbar_dw), axis=0) / self.x.shape[1]
    #self.grads['bias'] = dout.dot(dxbar_db).T / self.x.shape[1]
    #print("self.grads['weight']", self.grads['weight'].shape)
    #print("self.grads['bias']", self.grads['bias'].shape)

    dx = np.einsum('ij,ijk->ik', dout, dxbar_dx)
    '''
    #print("dx", dx.shape)
    
    #print("I:", dout)
    #print("O:", dx)

    self.grads['weight'] = np.dot(dout.T, self.x)
    self.grads['bias'] = np.sum(dout, axis = 0)
    self.grads['bias'] = self.grads['bias']#.reshape(self.grads['bias'].shape[0],1)
    dx = np.dot(dout, self.params['weight'])

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
    
    #print("x", x.shape)
    #print("out:", out.shape)

    #print("I:", x)
    #print("O:", out)


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

    dout = dout.T

    dx_dxbar = []
    
    for x_idx in range(self.x.shape[1]):
        dx_dxbar_one = np.zeros(shape=(self.x.shape[0], self.x.shape[0]))
        gtz_idx = np.argwhere(self.x[:,x_idx] > 0.0)
        dx_dxbar_one[gtz_idx,gtz_idx] = 1
        dx_dxbar += [dx_dxbar_one]

    dx_dxbar = np.array(dx_dxbar)
    
    dx = np.einsum('ij,ijk->ik', dout, dx_dxbar).T

    '''print("dout:", dout.shape)
    print("dx_dxbar:", dx_dxbar.shape)
    print("dx", dx.shape)'''

    #print("I:", dout)
    #print("O:", dx)


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

    #print(x)
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1).reshape(x.shape[0], 1)

    out = x_exp / sum_exp
    
    self.x = x
    self.out = out
    self.x_exp = x_exp
    self.sum_exp = sum_exp

    '''print("x_exp:", x_exp.shape)
    print("x:", x.shape)
    print("sum_exp:", sum_exp.shape)
    print("out:", out.shape)'''
    
    #print("I:", x)
    #print("O:", out)

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
    
    dout = dout
    '''print("dout:", dout.shape)
    print("x_exp:", self.x_exp.shape)
    print("sum_exp:", self.sum_exp.shape)
    print("self.x_exp * self.sum_exp:", (self.x_exp * self.sum_exp).shape)'''
    dx_dxbar = []
    for i in range(self.x.shape[0]):
        x_exp = self.x_exp[i,:]
        x_exp = x_exp.reshape(1, x_exp.shape[0])


        dx_dxbar += [(np.diag((x_exp * self.sum_exp[i,:])[0,:]) - x_exp.T.dot(x_exp)) / (self.sum_exp[i,:]) ** 2] 
    
    #print("diag:",np.diag(self.x_exp[:,i] * self.sum_exp[:,i]), x_exp.dot(x_exp.T))

    dx_dxbar = np.array(dx_dxbar)
     
    #dx = dout.dot(dx_dxbar)

    #print(dout, dx_dxbar)

    dx = np.einsum('ij,ijk->ik', dout, dx_dxbar)
    
    '''print("dout:", dout.shape)
    print("dx_dxbar:", dx_dxbar.shape)
    print("dx", dx.shape)'''

    #print("I:", dout)
    #print("dx_dxbar:", dx_dxbar)
    #print("x:", self.x)
    #print("O:", dx)

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

    #y = y.T

    #out = -np.log(x[:,np.argmax(y,axis = 1)])
    out = -np.mean(np.sum(np.log(x) * y, axis = 0))    
    #print("x:", x.shape)
    #print("y:", y.shape)
    #print("argmax: ", np.argmax(y,axis = 1).shape)
    #print("out:", out.shape)
 
    #print("Ix:", x)
    #print("Iy:", y)
    #print("O:", out)


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
    
    #print("-----------------Cross B----------------")

    dx = - y / x / x.shape[0]

    
    #print("x:", x.shape)
    #print("y:", y.shape)
    #print("dx", dx.shape)

    #print("Ix:", x)
    #print("Iy:", y)
    #print("O:", dx)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
