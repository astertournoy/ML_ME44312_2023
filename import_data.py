#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


path = os.getcwd() + '\ModeChoiceOptima.txt'
df = pd.read_csv(path, sep="\t")
# df.head()
# df.describe()

#implement a scatter plot for the data (see example in the slides)
# plt.scatter(df.ID, df.DestAct)
# plt.show()


# sort out choices of transport mode per ID
df.sort_index()











