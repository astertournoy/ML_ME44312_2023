import import_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = import_data.df


# ------------ STEP 1: import data for this question ------------
# DATASET1 NEEDED: TYPE OF COMMUNE
TC = df.TypeCommune

# DATASET2 NEEDED: PREFERRED MODE OF TRANSPORT
# We plot the data first to have an idea what it looks like

# We make a dictionary per ID, that says how often that ID chooses 0 (public transport),\
# 1 (private mode) or 2 (soft mode) AND that says

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
    # which mode is preferred?
    if max(count_0,count_1,count_2) == count_0:
        preferred_mode = 0
    elif max(count_0,count_1,count_2) ==  count_1:
        preferred_mode = 1
    elif max(count_0,count_1,count_2) == count_2:
        preferred_mode = 2
    if (count_0 == count_1 and count_0!= 0) or (count_2 == count_1 and count_2 != 0) or (count_2 == count_0 and count_0 != 0):
        preferred_mode = -2
    if count_0 == count_1 and count_2 == count_1:
        preferred_mode = -1
    # Get the value of TC for the current index value
    tc_value = sub_df['TypeCommune'].iloc[0]
    # Add the counts to the dictionary
    result_dict[idx] = {'public_transport': count_0, 'private_mode': count_1, 'soft_mode': count_2, \
                        'PM': preferred_mode, 'TC': tc_value}

print(result_dict)
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


# Now we want to check how many people live in the different types of communes to have a rough idea.
# Similarly we want to check how many people prefer what modes
# Initialize a counter variable to 0
count_tc_1 = 0
count_tc_2 = 0
count_tc_3 = 0
count_tc_4 = 0
count_tc_5 = 0
count_tc_6 = 0
count_tc_7 = 0
count_tc_8 = 0
count_tc_9 = 0
count_PM_0 = 0
count_PM_1 = 0
count_PM_2 = 0
count_PM_none = 0
count_PM_multiple = 0

# Loop through each key-value pair in the dictionary
for idx, values in result_dict.items():
    # Check if the value of 'TC' is equal to 1
    if values['TC'] == 1:
        # If so, increment the counter variable
        count_tc_1 += 1
    elif values['TC'] == 2:
        # If so, increment the counter variable
        count_tc_2 += 1
    elif values['TC'] == 3:
        # If so, increment the counter variable
        count_tc_3 += 1
    elif values['TC'] == 4:
        # If so, increment the counter variable
        count_tc_4 += 1
    elif values['TC'] == 5:
        # If so, increment the counter variable
        count_tc_5 += 1
    elif values['TC'] == 6:
        # If so, increment the counter variable
        count_tc_6 += 1
    elif values['TC'] == 7:
        # If so, increment the counter variable
        count_tc_7 += 1
    elif values['TC'] == 8:
        # If so, increment the counter variable
        count_tc_8 += 1
    elif values['TC'] == 9:
        # If so, increment the counter variable
        count_tc_9 += 1

for idx, values in result_dict.items():
    if values['PM'] == 0:
        count_PM_0 += 1
    elif values['PM'] == 1:
        count_PM_1 += 1
    elif values['PM'] == 2:
        count_PM_2 += 1
    elif values['PM'] == -1:
        count_PM_none += 1
    elif values['PM'] == -2:
        count_PM_multiple += 1


# print('centers', count_tc_1)
# print('suburban', count_tc_2)
# print('high-income', count_tc_3)
# print('periurban', count_tc_4)
# print('touristic', count_tc_5)
# print('industrial', count_tc_6)
# print('rural', count_tc_7)
# print('agri-mixed', count_tc_8)
# print('agricultural', count_tc_9)
# print('public transport', count_PM_0)
# print('private transport', count_PM_1)
# print('soft mode', count_PM_2)
# print('none preferred', count_PM_none)
# print('multiple preferred', count_PM_multiple)

# total_c = count_tc_1+count_tc_2+count_tc_8+count_tc_9+count_tc_6+count_tc_7+count_tc_4+count_tc_5+count_tc_3
# print(total)
# total_pm = count_PM_none + count_PM_multiple + count_PM_0 + count_PM_1 + count_PM_2
# total_useful = count_PM_0 + count_PM_1 + count_PM_2
# print(total_useful)


# ------------ STEP 2: data preparation ------------
# split training data and testing data







# ------------ STEP 3: Apply Maching Learning ------------










# ------------ STEP 4: Result Analysis ------------