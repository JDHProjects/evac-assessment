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

  def senseWall(self, pos):
    return(pos[0] <= 0 or pos[0] >= (self.YSIZE-1) or pos[1] <= 0 or pos[1] >= (self.XSIZE-1))

  def senseTail(self, pos):
    return pos in self.snake

  def leftWall(self):
    return self.senseWall([self.snake[0][0],self.snake[0][1]-1])

  def leftTail(self):
    return self.senseTail([self.snake[0][0],self.snake[0][1]-1])

  def leftFood(self):
    return self.food[1] < self.snake[0][1]

  def rightWall(self):
    return self.senseWall([self.snake[0][0],self.snake[0][1]+1])

  def rightTail(self):
    return self.senseTail([self.snake[0][0],self.snake[0][1]+1])

  def rightFood(self):
    return self.food[1] > self.snake[0][1]

  def upWall(self):
    return self.senseWall([self.snake[0][0]-1,self.snake[0][1]])

  def upTail(self):
    return self.senseTail([self.snake[0][0]-1,self.snake[0][1]])

  def upFood(self):
    return self.food[0] < self.snake[0][0]

  def downWall(self):
    return self.senseWall([self.snake[0][0]+1,self.snake[0][1]])

  def downTail(self):
    return self.senseTail([self.snake[0][0]+1,self.snake[0][1]])

  def downFood(self):
    return self.food[0] > self.snake[0][0]

  def getInputs(self):
      return [  self.leftWall(), self.leftTail(), self.leftFood(),
                self.rightWall(), self.rightTail(), self.rightFood(),
                self.upWall(), self.upTail(), self.upFood(),
                self.downWall(), self.downTail(), self.downFood(),
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

  toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)


  stats = tools.Statistics(key=lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)


  logbook = tools.Logbook()
  NGEN = 500
  CXPB = 0
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

    offspring = toolbox.select(pop, len(pop)//2)
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

    pop[:] = offspring + tools.selBest(pop, len(pop) - (len(pop)//2))
    record = stats.compile(pop)
    logbook.record(gen=g, **record)
    print("Score Max    : " + str(np.max(scores)))
    print("Score Mean   : " + str(np.mean(scores)))
    print("Fitness max  : "+str(record['max']))
    print("Fitness mean : "+str(record['avg']))
  return pop, logbook

def genNetwork():
  numInputNodes = 12
  numHiddenNodes1 = 12
  numHiddenNodes2 = 12
  numOutputNodes = 4

  IND_SIZE = ((numInputNodes+1) * numHiddenNodes1) + (numHiddenNodes1 *
                                                      numHiddenNodes2) + (numHiddenNodes2 * numOutputNodes)

  return neuralNetwork.NeuralNetwork(numInputNodes, numHiddenNodes1, numHiddenNodes2, numOutputNodes), IND_SIZE

 
if __name__ == "__main__":

  snake_game = currentSnake(XSIZE, YSIZE)

  network, IND_SIZE = genNetwork()

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