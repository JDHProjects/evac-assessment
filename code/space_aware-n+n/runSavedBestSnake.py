from deap import creator
from deap import base

import displayGame
import runGame
import neuralNetwork
import snake

import pickle

# Number of grid cells in each direction (do not change this)
XSIZE = YSIZE = 16

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def genNetwork():
  numInputNodes = 8
  numHiddenNodes = 12
  numOutputNodes = 4

  IND_SIZE = ((numInputNodes+1) * numHiddenNodes) +  + (numHiddenNodes * numOutputNodes)

  return neuralNetwork.NeuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes), IND_SIZE

if __name__ == "__main__":

  snake_game = snake.snake(XSIZE, YSIZE)

  network, IND_SIZE = genNetwork()

  with open ("data/data-1.ind", 'rb') as readFile:
    bestInd = pickle.load(readFile)

  network.setWeightsLinear(bestInd)

  runGame.run_game(network, displayGame.DisplayGame(XSIZE, YSIZE), snake_game, headless=False)