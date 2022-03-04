import numpy as np

class NeuralNetwork(object):
  def __init__(self, numInput, numHidden, numOutput):
    self.fitness = 0
    self.numInput = numInput + 1 # Add bias node to inputs
    self.numHidden = numHidden
    self.numOutput = numOutput

    self.wh = np.random.randn(self.numHidden, self.numInput) 
    self.wo = np.random.randn(self.numOutput, self.numHidden)

    self.ReLU = lambda x : max(0,x)

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def feedForward(self, inputs):
    inputsBias = inputs[:]
    inputsBias.insert(len(inputs), 1)

    h1 = np.dot(self.wh, inputsBias)
    h1 = [self.ReLU(x) for x in h1]

    output = np.dot(self.wo, h1)
    return self.softmax(output)

  def getWeightsLinear(self):
    flat_wh = list(self.wh.flatten())
    flat_wo = list(self.wo.flatten())
    return( flat_wh + flat_wo )

  def setWeightsLinear(self, Wgenome):
    numWeights_IH = self.numHidden * (self.numInput)
    self.wh = np.array(Wgenome[:numWeights_IH])
    self.wh = self.wh.reshape((self.numHidden, self.numInput))
    self.wo = np.array(Wgenome[numWeights_IH:])
    self.wo = self.wo.reshape((self.numOutput, self.numHidden))
