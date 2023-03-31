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
    result_dict[idx] = {'public_transport': count_0, 'private_mode': count_1, 'soft_mode': count_2}



# # Define your dictionary with data
# data = {
#     'id1': {'public_transport': 10, 'private_mode': 20, 'soft_mode': 5},
#     'id2': {'public_transport': 15, 'private_mode': 25, 'soft_mode': 10},
#     'id3': {'public_transport': 20, 'private_mode': 30, 'soft_mode': 15},
#     'id4': {'public_transport': 25, 'private_mode': 35, 'soft_mode': 20},
#     'id5': {'public_transport': 30, 'private_mode': 40, 'soft_mode': 25}
# }
#
# # Create an empty list to store the public_transport values
# public_transport_values = []
#
# # Loop through each id in the dictionary and extract the public_transport value
# for id, values in data.items():
#     public_transport_values.append(values['public_transport'])
#
# # Print the list of public_transport values
# print(public_transport_values)



# Extract the data for each axis
# x = result_dict('public_transport')
# y = result_dict['private_mode']
# z = result_dict['soft_mode']

# Create the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
#
# # Label the axes
# ax.set_xlabel('Public Transport')
# ax.set_ylabel('Private Mode')
# ax.set_zlabel('Soft Mode')
#
# # Show the plot
# plt.show()



