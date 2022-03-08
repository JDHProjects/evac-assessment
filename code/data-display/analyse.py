from cProfile import label
import matplotlib.pyplot as plt
import pickle
import numpy as np

def generateBoxplot(data, names):

  fig = plt.figure(figsize =(10, 7)) 
  ax = fig.add_axes([0, 0, 1, 1]) 
  print(names)
  ax.set_xticklabels(names) 

  ax.set_xlabel('Condition')
  ax.set_ylabel('Score')

  ax.boxplot(data)
  plt.show()
  fig.show()

def generateGraph(gen, stats, xLabel, yLabel, title="", filename=""):
  plt.figure()
  if(title != ""):
    plt.title(title)
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
    end = len(gens)
    mins.append(logbook.select("min")[-1])
    maxs.append(logbook.select("max")[-1])
    avgs.append(logbook.select("avg")[-1])
    stds.append(logbook.select("std")[-1])

  return {"gen": gens, "avg": avgs, "min": np.mean(mins, axis=0), "max": np.mean(maxs, axis=0), "std": np.mean(stds, axis=0), "label": name, "colour": colour }


if __name__ == "__main__":

  #with open ("../local-n+n/data/data-1.stats", 'rb') as readFile:
  #  stats = pickle.load(readFile)

  #generateScoreGraph(stats)
  stats1  = getAverageData("../local-standard/data", "local-standard", "blue")
  stats2  = getAverageData("../local-n+n/data", "local-standard", "blue")
  stats3  = getAverageData("../space_aware-standard/data", "space_aware", "blue")
  stats4  = getAverageData("../space_aware-n+n/data", "space_aware-n+n", "orange")
  from scipy.stats import mannwhitneyu
  stat, p = mannwhitneyu(stats1["avg"], stats2["avg"])
  print('Statistics=%.3f, p=%.10f' % (stat, p))
  print(stats1["avg"])
  generateBoxplot([stats1["avg"],stats2["avg"],stats3["avg"],stats4["avg"]],["A","B","C","D"])
  #print(generateGraph(stats1["gen"], [stats1,stats2], "Generations", "Mean Score"))


