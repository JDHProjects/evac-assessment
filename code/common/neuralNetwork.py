import numpy as np

class NeuralNetwork(object):
  def __init__(self, numInput, numHidden1, numHidden2, numOutput):
    self.fitness = 0
    self.numInput = numInput + 1  # Add bias node from input to hidden layer 1 only
    # Feel free to adapt the code to add more biases if you wish
    self.numHidden1 = numHidden1
    self.numHidden2 = numHidden2
    self.numOutput = numOutput

    self.w_i_h1 = np.random.randn(self.numHidden1, self.numInput)
    self.w_h1_h2 = np.random.randn(self.numHidden2, self.numHidden1)
    self.w_h2_o = np.random.randn(self.numOutput, self.numHidden2)

    self.ReLU = lambda x: max(0, x)

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def feedForward(self, inputs):
    inputsBias = inputs[:]
    inputsBias.insert(len(inputs), 1)             # Add bias input

    # feed input to hidden layer 1
    h1 = np.dot(self.w_i_h1, inputsBias)
    h1 = [self.ReLU(x) for x in h1]              # Activate hidden layer1

    # feed layer 1 to hidden layer 2
    h2 = np.dot(self.w_h1_h2, h1)
    h2 = [self.ReLU(x) for x in h2]              # Activate hidden layer 2

    output = np.dot(self.w_h2_o, h2)             # feed to output layer
    # output = [self.sigmoid(x) for x in output]   # Activate output layer
    return self.softmax(output)

  def getWeightsLinear(self):
    flat_w_i_h1 = list(self.w_i_h1.flatten())
    flat_w_h1_h2 = list(self.w_h1_h2.flatten())
    flat_w_h2_o = list(self.w_h2_o.flatten())
    return(flat_w_i_h1 + flat_w_h1_h2 + flat_w_h2_o)

  def setWeightsLinear(self, Wgenome):
    numWeights_I_H1 = self.numHidden1 * self.numInput
    numWeights_H1_H2 = self.numHidden2 * self.numHidden1
    numWeights_H2_O = self.numOutput * self.numHidden2

    self.w_i_h1 = np.array(Wgenome[:numWeights_I_H1])
    self.w_i_h1 = self.w_i_h1.reshape((self.numHidden1, self.numInput))

    self.w_h1_h2 = np.array(
        Wgenome[numWeights_I_H1:(numWeights_H1_H2+numWeights_I_H1)])
    self.w_h1_h2 = self.w_h1_h2.reshape((self.numHidden2, self.numHidden1))

    self.w_h2_o = np.array(Wgenome[(numWeights_H1_H2+numWeights_I_H1):])
    self.w_h2_o = self.w_h2_o.reshape((self.numOutput, self.numHidden2))
