from deap import base
from deap import creator
from deap import tools
from datetime import datetime
import random
import common.neuralNetwork as neuralNetwork
import common.baseSnake as baseSnake
import common.displayGame as displayGame
import common.runGame as runGame
import numpy as np
import pickle

# Number of grid cells in each direction (do not change this)
XSIZE = YSIZE = 16

random.seed(datetime.now())  # Set a random seed from the clock
  
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class currentSnake(baseSnake.snake):

  def starved(self, timeSinceEaten):
    if(10*len(self.snake) < timeSinceEaten):
      return True
    return False

  def getAheadLocation(self):
    return [self.snake[0][0] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1),
                  self.snake[0][1] + (self.snake_direction == "left" and -1) + (self.snake_direction == "right" and 1)]

  def getBehindLocation(self):
    return [self.snake[0][0] + (self.snake_direction == "down" and -1) + (self.snake_direction == "up" and 1),
                  self.snake[0][1] + (self.snake_direction == "left" and 1) + (self.snake_direction == "right" and -1)]
  
  def getLeftLocation(self):
    return [self.snake[0][0] + (self.snake_direction == "left" and 1) + (self.snake_direction == "right" and -1),
                  self.snake[0][1] + (self.snake_direction == "down" and -1) + (self.snake_direction == "up" and 1)]
  
  def getRightLocation(self):
    return [self.snake[0][0] + (self.snake_direction == "left" and -1) + (self.snake_direction == "right" and 1),
                  self.snake[0][1] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1)]

  # y,x
  # 0 | 1
  # --s--
  # 2 | 3
  def senseRelativeFoodQuadrent(self):
    #up
    if (self.snake[0][0] >= self.food[0]):
      if (self.snake[0][1] >= self.food[1]):
        return 0
      return 1
    #down
    if (self.snake[0][1] >= self.food[1]):
      return 2
    return 3

  def foodLeft(self):
    return (self.senseRelativeFoodQuadrent() == 0) or (self.senseRelativeFoodQuadrent() == 2)

  def foodRight(self):
    return (self.senseRelativeFoodQuadrent() == 1) or (self.senseRelativeFoodQuadrent() == 3)

  def foodAhead(self):
    return (self.senseRelativeFoodQuadrent() == 0) or (self.senseRelativeFoodQuadrent() == 1)

  def foodBehind(self):
    return (self.senseRelativeFoodQuadrent() == 2) or (self.senseRelativeFoodQuadrent() == 3)

  def senseWall(self, pos):
    return(pos[0] <= 0 or pos[0] >= (YSIZE-1) or pos[1] <= 0 or pos[1] >= (XSIZE-1))

  def senseTail(self, pos):
    return pos in self.snake

  def aheadCollision(self):
    ahead = self.getAheadLocation()
    return self.senseWall(ahead) or self.senseTail(ahead)

  def behindCollision(self):
    behind = self.getBehindLocation()
    return self.senseWall(behind) or self.senseTail(behind)

  def leftCollision(self):
    left = self.getLeftLocation()
    return self.senseWall(left) or self.senseTail(left)

  def rightCollision(self):
    right = self.getRightLocation()
    return self.senseWall(right) or self.senseTail(right)

  def getInputs(self):
      return [  snake_game.foodLeft(), snake_game.leftCollision(),
                snake_game.foodRight(), snake_game.rightCollision(),
                snake_game.aheadCollision(), snake_game.foodAhead(),
                snake_game.behindCollision(), snake_game.foodBehind()
                ]

  def getFitness(self, score, timeAlive):
      return (score ** 3) * timeAlive

def train(network, snake_game, IND_SIZE):

  def evaluate(indiv, network):
    fitness = 0
    network.setWeightsLinear(indiv)
    fitness, score = runGame.run_game(network, None, snake_game, headless=True)
    return ((fitness,), score)

  toolbox = base.Toolbox()
  toolbox.register("attr_float", random.uniform, -1.0, 1.0)
  toolbox.register("individual", tools.initRepeat, creator.Individual,
                  toolbox.attr_float, n=IND_SIZE)

  toolbox.register("mate", tools.cxOnePoint)

  toolbox.register("evaluate", evaluate)


  toolbox.register("select", tools.selTournament, tournsize=3)

  toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.2)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)


  stats = tools.Statistics(key=lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)


  logbook = tools.Logbook()


  NGEN = 1000
  CXPB = 0.0
  MUTPB = 1
  POP = 200

  pop = toolbox.population(n=POP)
  fitnesses = []
  scores = []
  for indiv in pop:
    fitness, score = toolbox.evaluate(indiv, network)
    fitnesses.append(fitness)
    scores.append(score)
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


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
    fitnesses = []
    scores = []
    for indiv in invalid_ind:
      fitness, score = toolbox.evaluate(indiv, network)
      fitnesses.append(fitness)
      scores.append(score)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    pop[:] = offspring
    record = stats.compile(pop)
    logbook.record(gen=g, **record)
    print("Score Max    : " + str(np.max(scores)))
    print("Score Mean   : " + str(np.mean(scores)))
    print("Fitness max  : "+str(record['max']))
    print("Fitness mean : "+str(record['avg']))
  return pop, logbook

 
if __name__ == "__main__":

  snake_game = currentSnake(XSIZE, YSIZE)

  numInputNodes = 8
  numHiddenNodes1 = 10
  numHiddenNodes2 = 10
  numOutputNodes = 4

  IND_SIZE = ((numInputNodes+1) * numHiddenNodes1) + (numHiddenNodes1 *
                                                      numHiddenNodes2) + (numHiddenNodes2 * numOutputNodes)

  network = neuralNetwork.NeuralNetwork(numInputNodes, numHiddenNodes1, numHiddenNodes2, numOutputNodes)

  #'''
  (pop, logbook) = train(network, snake_game, IND_SIZE)
  
  with open("dump.pop", 'wb') as writeFile:
    pickle.dump(pop, writeFile)

  bestInd = tools.selBest(pop, 1)[0]

  with open("dump.ind", 'wb') as writeFile:
    pickle.dump(bestInd, writeFile)

  network.setWeightsLinear(bestInd)

  runGame.run_game(network, displayGame.DisplayGame(XSIZE, YSIZE), snake_game, headless=False)

  '''
  with open ("dump.ind", 'rb') as readFile:
    bestInd = pickle.load(readFile)

  network.setWeightsLinear(bestInd)

  runGame.run_game(network, displayGame.DisplayGame(XSIZE, YSIZE), snake_game, headless=False)
  #'''