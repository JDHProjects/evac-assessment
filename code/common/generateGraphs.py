import matplotlib.pyplot as plt
import numpy as np

def generateGraph(gen, xLabel, yLabel, title="", filename="", avg=[], min=[], max=[]):
  if (avg+min+max == []):
    print("No data to plot")
    return
  plt.figure()
  if(title != ""):
    plt.title(title)
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  if (avg != []):
    plt.plot(gen, avg)
  if (min != []):
    plt.plot(gen, min)
  if (max != []):
    plt.plot(gen, max)
  
  if(filename!=""):
    plt.savefig(filename)
    return
  plt.show()

def generateFitnessGraph(stats, filename="", title="", xLabel="Generation", yLabel="Fitness"):
  logbook = stats["logbook"]
  gen = logbook.select("gen")
  min = logbook.select("min")
  max = logbook.select("max")
  avg = logbook.select("avg")
  return generateGraph(gen, xLabel, yLabel, title, filename, avg, min, max)

def generateScoreGraph(stats, filename="", title="", xLabel="Generation", yLabel="Score"):
  logbook = stats["logbook"]
  scores = stats["scores"]
  gen = logbook.select("gen")
  min = np.min(scores, axis=1)
  max = np.max(scores, axis=1)
  avg = np.mean(scores, axis=1)
  return generateGraph(gen, xLabel, yLabel, title, filename, avg, min, max)
