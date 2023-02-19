#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


path = os.getcwd() + '\ModeChoiceOptima.txt'
data = pd.read_csv(path, sep="\t")
# data = pd.read_csv(path, header=None, names=['ID', 'DestAct'])
data.head()
data.describe()

#implement a scatter plot for the data (see example in the slides)
plt.scatter(data.ID, data.DestAct)
plt.show()
