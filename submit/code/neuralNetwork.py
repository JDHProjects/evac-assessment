"""
  Neural Network class. Code overview can be found in report appendix
"""

import numpy as np

class NeuralNetwork(object):
  def __init__(self, numInput, numHidden, numOutput):
    """
    Setup initial state of neural network
    args:
      int - numInput - number of input nodes
      int - numHidden - number of hidden nodes
      int - numOutput - number of output nodes
    return:
      none
    """

    self.fitness = 0
    self.numInput = numInput + 1 # Add bias node to inputs
    self.numHidden = numHidden
    self.numOutput = numOutput

    self.wh = np.random.randn(self.numHidden, self.numInput) 
    self.wo = np.random.randn(self.numOutput, self.numHidden)

    self.ReLU = lambda x : max(0,x)

  def softmax(self, x):
    """
    Softmax function to apply to output layer
    args:
      list - x - output layer values
    return:
      list - softmax of x
    """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def feedForward(self, inputs):
    """
    Run inputs through the neural network
    args:
      list - inputs - values to input to neural network
    return:
      list - softmax of output layer
    """

    inputsBias = inputs[:]
    inputsBias.insert(len(inputs), 1)

    h1 = np.dot(self.wh, inputsBias)
    h1 = [self.ReLU(x) for x in h1]

    output = np.dot(self.wo, h1)
    return self.softmax(output)

  def getWeightsLinear(self):
    """
    Get weights of network
    args:
      none
    return:
      tuple - network weights of hidden and output layers
    """

    flat_wh = list(self.wh.flatten())
    flat_wo = list(self.wo.flatten())
    return( flat_wh + flat_wo )

  def setWeightsLinear(self, Wgenome):
    """
    Update weights of neural network
    args:
      list - Wgenome - weights of individual
    return:
      none
    """

    numWeights_IH = self.numHidden * (self.numInput)
    self.wh = np.array(Wgenome[:numWeights_IH])
    self.wh = self.wh.reshape((self.numHidden, self.numInput))
    self.wo = np.array(Wgenome[numWeights_IH:])
    self.wo = self.wo.reshape((self.numOutput, self.numHidden))
