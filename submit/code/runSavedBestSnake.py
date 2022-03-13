"""
  Used to display a saved snake playing the snake game. Code overview can be found in report appendix
"""

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
  # setup network

  numInputNodes = 12
  numHiddenNodes = 16
  numOutputNodes = 4

  IND_SIZE = ((numInputNodes+1) * numHiddenNodes) +  + (numHiddenNodes * numOutputNodes)

  return neuralNetwork.NeuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes), IND_SIZE

if __name__ == "__main__":
  # initialize snake game
  snake_game = snake.snake(XSIZE, YSIZE)

  # initialize neural network
  network, IND_SIZE = genNetwork()

  # load saved best snake individual
  with open ("bestSnake.ind", 'rb') as readFile:
    bestInd = pickle.load(readFile)

  # set weights of individual
  network.setWeightsLinear(bestInd)

  # run snake game with best individual
  runGame.run_game(network, displayGame.DisplayGame(XSIZE, YSIZE), snake_game, headless=False)