import import_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = import_data.df


# ------------ STEP 1: import data for this question ------------
# DATASET1 NEEDED: PREFERRED MODE OF TRANSPORT
# We plot the data first to have an idea what it looks like

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
# print(len(result_dict))

# Create an empty list to store the public_transport values
public_transport_values = []
private_mode_values = []
soft_mode_values = []

# add values of data to lists
for id, values in result_dict.items():
    public_transport_values.append(values['public_transport'])
    private_mode_values.append(values['private_mode'])
    soft_mode_values.append(values['soft_mode'])


# plot data
x = public_transport_values
y = private_mode_values
z = soft_mode_values

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Label the axes
ax.set_xlabel('Public Transport')
ax.set_ylabel('Private Mode')
ax.set_zlabel('Soft Mode')

# Show the plot
plt.show()





# ------------ STEP 2: data preparation ------------
# add labels to the modes of transports
