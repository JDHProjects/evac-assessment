import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
from scipy.stats import mannwhitneyu

def generateBoxplot(data, names):

  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
  ax.boxplot(data)
  ax.set_title('Average of 10 Runs At Generation 500, Performance of Each Snake Algorithm')
  ax.set_xlabel('Algorithm')
  ax.set_xticklabels(names) 
  ax.set_ylabel('Score')

  plt.show()

def generateGraph(gen, stats, xLabel, yLabel, title="", filename=""):
  plt.figure()
  if(title != ""):
    plt.title(title)
  #else:
    #plt.title('Average of 10 Runs, evolution of\n'+stats[0]["label"]+' vs '+stats[1]["label"])
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  for stat in stats:
    if ("avg" in stat):
      avgLabel = "Mean"
      if ("label" in stat):
        avgLabel = stat["label"]
      avgColour = "Mean"
      if ("colour" in stat):
        avgColour = stat["colour"]
      plt.plot(gen, stat["avg"], label=avgLabel, color=avgColour)
      if ("std" in stat):
        plt.fill_between(gen, stat["avg"]+stat["std"], stat["avg"]-stat["std"], facecolor=avgColour, alpha=0.5)
    if(len(stats) < 2):
      if ("min" in stat):
        plt.plot(gen, stat["min"], label='Minimum')
      if ("max" in stat):
        plt.plot(gen, stat["max"], label='Maximum')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
  if(filename!=""):
    plt.savefig(filename)
    return
  plt.show()

def generateFitnessGraph(stats, filename="", title="", xLabel="Generation", yLabel="Fitness"):
  logbook = stats["fitnessLogbook"]
  gen = logbook.select("gen")
  stat = {
    "min": logbook.select("min"),
    "max": logbook.select("max"),
    "avg": logbook.select("avg"),
  }
  return generateGraph(gen, [stat], xLabel, yLabel, title, filename)

def generateScoreGraph(stats, filename="", title="", xLabel="Generation", yLabel="Score"):
  logbook = stats["scoreLogbook"]
  gen = logbook.select("gen")
  stat = {
    "min": logbook.select("min"),
    "max": logbook.select("max"),
    "avg": logbook.select("avg"),
  }
  return generateGraph(gen, [stat], xLabel, yLabel, title, filename)

def getAverageData(dataLocation, name, colour, filename="", title="", xLabel="Generation", yLabel="Score"):
  ret = getAllData(dataLocation, name, colour, filename, title, xLabel, yLabel)

  return {"gen": ret["gen"], "avg": np.mean(ret["avg"], axis=0), "min": np.mean(ret["min"], axis=0), "max": np.mean(ret["max"], axis=0), "std": np.mean(ret["std"], axis=0), "label": name, "colour": colour }

def getLastGenData(dataLocation, name, colour, filename="", title="", xLabel="Generation", yLabel="Score"):
  ret = getAllData(dataLocation, name, colour, filename, title, xLabel, yLabel)

  return {"gen": ret["gen"], "avg": ret["avg"][:,-1], "min": ret["min"][:,-1], "max": ret["max"][:,-1], "std": ret["std"][:,-1], "label": name, "colour": colour }

def getAllData(dataLocation, name, colour, filename="", title="", xLabel="Generation", yLabel="Score"):
  avgs = []
  gens = None
  mins = []
  maxs = [] 
  stds = []
  for i in range(1,11):
    with open (dataLocation+"/data-"+str(i)+".stats", 'rb') as readFile:
      stats = pickle.load(readFile)
    logbook = stats["scoreLogbook"]
    gens = logbook.select("gen")
    mins.append(logbook.select("min"))
    maxs.append(logbook.select("max"))
    avgs.append(logbook.select("avg"))
    stds.append(logbook.select("std"))

  return {"gen": np.array(gens), "avg": np.array(avgs), "min": np.array(mins), "max": np.array(maxs), "std": np.array(stds), "label": name, "colour": colour }


def getCohensD(meanDiff, std1, std2):
  N=10
  offset = ((N-3)/(N-2.25) + math.sqrt((N-2)/2)) 

  return (meanDiff/(math.sqrt(((std1**2) + (std2**2)) /2 ))) * offset


def getAverageUAndP(stats1, stats2):
  allP = []
  allU = []
  allMeanDiff = []
  lenGen = len(stats1["gen"])
  for i in range(0,lenGen):
    u, p = mannwhitneyu(stats1["avg"][:,i], stats2["avg"][:,i])
    allP.append(p)
    allU.append(min(u, (len(stats1["avg"][:,i]) * len(stats2["avg"][:,i]))-u))
    allMeanDiff.append(abs(np.mean(stats1["avg"][:,i]) - np.mean(stats2["avg"][:,i])))

  return np.mean(allU), np.mean(allP), np.mean(allMeanDiff)



if __name__ == "__main__":
  
  #with open ("../local-n+n/data/data-1.stats", 'rb') as readFile:
  #  stats = pickle.load(readFile)

  #generateScoreGraph(stats)
  stats1  = getAllData("../local-standard/data", "local-standard", "blue")
  stats2  = getAllData("../local-n+n/data", "local-n+n", "orange")
  stats3  = getAllData("../space_aware-standard/data", "space_aware-standard", "red")
  stats4  = getAllData("../space_aware-n+n/data", "space_aware-n+n", "green")
  stat, p, d = getAverageUAndP(stats1,stats2)
  print('Local: U=%.1f, p=%.6f, Mean diff=%.6f' % (stat, p, d))
  stat, p, d = getAverageUAndP(stats1,stats3)
  print('Standard: U=%.1f, p=%.6f, Mean diff=%.6f' % (stat, p, d))
  stat, p, d = getAverageUAndP(stats3,stats4)
  print('Space Aware: U=%.1f, p=%.6f, Mean diff=%.6f' % (stat, p, d))
  stat, p, d = getAverageUAndP(stats2,stats4)
  print('N+N: U=%.1f, p=%.6f, Mean diff=%.6f' % (stat, p, d))

  #generateBoxplot([stats1["avg"][:,-1],stats2["avg"][:,-1],stats3["avg"],stats4["avg"]],[stats1["label"],stats2["label"],stats3["label"],stats4["label"]])
  #print(generateGraph(stats1["gen"], [stats1,stats2], "Generations", "Mean Score", filename=""))


