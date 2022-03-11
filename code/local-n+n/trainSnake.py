from deap import base
from deap import creator
from deap import tools
from datetime import datetime
import random
import neuralNetwork
import snake
import runGame
import numpy as np
import pickle

# Number of grid cells in each direction (do not change this)
XSIZE = YSIZE = 16

random.seed(datetime.now())  # Set a random seed from the clock

def train(network, snake_game, IND_SIZE):

  def evaluate(indiv, network):
    fitness = 0
    network.setWeightsLinear(indiv)
    fitness, score = runGame.run_game(network, None, snake_game, headless=True)
    return ((fitness,), score)

  toolbox = base.Toolbox()
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax, score=0)

  toolbox.register("attr_float", random.uniform, -1.0, 1.0)
  toolbox.register("individual", tools.initRepeat, creator.Individual,
                  toolbox.attr_float, n=IND_SIZE)

  toolbox.register("mate", tools.cxOnePoint)

  toolbox.register("evaluate", evaluate)

  toolbox.register("select", tools.selTournament, tournsize=3)

  toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)


  fitnessStats = tools.Statistics(key=lambda ind: ind.fitness.values)
  fitnessStats.register("avg", np.mean)
  fitnessStats.register("std", np.std)
  fitnessStats.register("min", np.min)
  fitnessStats.register("max", np.max)

  fitnessLogbook = tools.Logbook()

  scoreStats = tools.Statistics(key=lambda ind: ind.score)
  scoreStats.register("avg", np.mean)
  scoreStats.register("std", np.std)
  scoreStats.register("min", np.min)
  scoreStats.register("max", np.max)

  scoreLogbook = tools.Logbook()

  NGEN = 500
  CXPB = 0.0
  MUTPB = 1
  POP = 200

  pop = toolbox.population(n=POP)
  for indiv in pop:
    fitness, score = toolbox.evaluate(indiv, network)
    indiv.fitness.values = fitness
    indiv.score = score

  for g in range(NGEN):
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for indiv in invalid_ind:
      fitness, score = toolbox.evaluate(indiv, network)
      indiv.fitness.values =fitness
      indiv.score = score

    pop[:] = tools.selBest(pop, len(pop) - (len(pop)//2)) + tools.selBest(offspring, len(pop)//2)
    fitnessRecord = fitnessStats.compile(pop)
    fitnessLogbook.record(gen=g, **fitnessRecord)
    scoreRecord = scoreStats.compile(pop)
    scoreLogbook.record(gen=g, **scoreRecord)
    print("Score Max    : " + str(scoreRecord["max"]))
    print("Score Mean   : " + str(scoreRecord["avg"]))
    print("Fitness max  : "+str(fitnessRecord['max']))
    print("Fitness mean : "+str(fitnessRecord['avg']))

  return pop, {"fitnessLogbook": fitnessLogbook, "scoreLogbook": scoreLogbook}

def genNetwork():
  numInputNodes = 12
  numHiddenNodes = 16
  numOutputNodes = 4

  IND_SIZE = ((numInputNodes+1) * numHiddenNodes) +  + (numHiddenNodes * numOutputNodes)

  return neuralNetwork.NeuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes), IND_SIZE

if __name__ == "__main__":

  snake_game = snake.snake(XSIZE, YSIZE)

  network, IND_SIZE = genNetwork()

  (pop, stats) = train(network, snake_game, IND_SIZE)

  filename = "data/data-1"
  with open(filename+".pop", 'wb') as writeFile:
    pickle.dump(pop, writeFile)

  bestInd = tools.selBest(pop, 1)[0]

  with open(filename+".ind", 'wb') as writeFile:
    pickle.dump(bestInd, writeFile)

  with open(filename+".stats", 'wb') as writeFile:
    pickle.dump(stats, writeFile)

  network.setWeightsLinear(bestInd)