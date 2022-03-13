"""
  Used to train a snake and save data to file. Code overview can be found in report appendix
"""

from deap import base
from deap import creator
from deap import tools

import neuralNetwork
import snake
import runGame
import numpy as np

from datetime import datetime
import random
import pickle

# Number of grid cells in each direction (do not change this)
XSIZE = YSIZE = 16

random.seed(datetime.now())  # Set a random seed from the clock

def train(network, snake_game, IND_SIZE):

  def evaluate(indiv, network):
    # run game for snake, return fitness and score
    fitness = 0
    network.setWeightsLinear(indiv)
    fitness, score = runGame.run_game(network, None, snake_game, headless=True)
    return ((fitness,), score)

  # setup toolbox
  toolbox = base.Toolbox()
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  # add score attribute to individual
  creator.create("Individual", list, fitness=creator.FitnessMax, score=0)

  toolbox.register("attr_float", random.uniform, -1.0, 1.0)
  toolbox.register("individual", tools.initRepeat, creator.Individual,
                  toolbox.attr_float, n=IND_SIZE)

  toolbox.register("mate", tools.cxOnePoint)

  toolbox.register("evaluate", evaluate)

  toolbox.register("select", tools.selTournament, tournsize=3)

  toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  # fitness logbook setup
  fitnessStats = tools.Statistics(key=lambda ind: ind.fitness.values)
  fitnessStats.register("avg", np.mean)
  fitnessStats.register("std", np.std)
  fitnessStats.register("min", np.min)
  fitnessStats.register("max", np.max)

  fitnessLogbook = tools.Logbook()

  # score logbook setup
  scoreStats = tools.Statistics(key=lambda ind: ind.score)
  scoreStats.register("avg", np.mean)
  scoreStats.register("std", np.std)
  scoreStats.register("min", np.min)
  scoreStats.register("max", np.max)

  scoreLogbook = tools.Logbook()

  # constants to tweak evolution with
  NGEN = 500
  CXPB = 0.0
  MUTPB = 1
  POP = 200

  # generate initial population
  pop = toolbox.population(n=POP)
  for indiv in pop:
    fitness, score = toolbox.evaluate(indiv, network)
    indiv.fitness.values = fitness
    indiv.score = score

  # run all generations
  for g in range(NGEN):
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # crossover code (found to make evolution worse for this implementation)
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    # mutate offspring
    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # run snake game on mutated individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for indiv in invalid_ind:
      fitness, score = toolbox.evaluate(indiv, network)
      indiv.fitness.values =fitness
      indiv.score = score
    
    # set population to offspring
    pop[:] = offspring
    # setup logbook
    fitnessRecord = fitnessStats.compile(pop)
    fitnessLogbook.record(gen=g, **fitnessRecord)
    scoreRecord = scoreStats.compile(pop)
    scoreLogbook.record(gen=g, **scoreRecord)

    # output data each generation
    print("Score Max    : " + str(scoreRecord["max"]))
    print("Score Mean   : " + str(scoreRecord["avg"]))
    print("Fitness max  : "+str(fitnessRecord['max']))
    print("Fitness mean : "+str(fitnessRecord['avg']))

  return pop, {"fitnessLogbook": fitnessLogbook, "scoreLogbook": scoreLogbook}

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

  # train population
  (pop, stats) = train(network, snake_game, IND_SIZE)

  # save snake data
  filename = "snake"
  with open(filename+".pop", 'wb') as writeFile:
    pickle.dump(pop, writeFile)

  bestInd = tools.selBest(pop, 1)[0]

  with open(filename+".ind", 'wb') as writeFile:
    pickle.dump(bestInd, writeFile)

  with open(filename+".stats", 'wb') as writeFile:
    pickle.dump(stats, writeFile)