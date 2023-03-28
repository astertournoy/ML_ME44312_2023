#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


path = os.getcwd() + '\ModeChoiceOptima.txt'
df = pd.read_csv(path, sep="\t")
# df.head()
# df.describe()

#implement a scatter plot for the data (see example in the slides)
# plt.scatter(df.ID, df.DestAct)
# plt.show()


# sort out choices of transport mode per ID
df.sort_index()




# We make a dictionary per ID, that says how often that ID chooses 0 (public transport),\
# 1 (private mode) or 2 (soft mode)


# Create an empty dictionary to store the results
result_dict = {}

# Loop through each unique index value
for idx in df['ID'].unique():
    # Subset the dataframe for the current index value
    sub_df = df[df['ID'] == idx]

    # Count the occurrences of 0, 1, and 2 in the 'Choice' column
    count_0 = (sub_df['Choice'] == 0).sum()
    count_1 = (sub_df['Choice'] == 1).sum()
    count_2 = (sub_df['Choice'] == 2).sum()

    # Add the counts to the dictionary
    result_dict[idx] = {'count_0': count_0, 'count_1': count_1, 'count_2': count_2}


# print(result_dict)


