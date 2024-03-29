import import_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
df = import_data.df
import pandas as pd



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

# print(result_dict)
# print(len(result_dict))



# ANALYSE DATA - plotting and printing
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


#Cleaning and creating final dataframe
MOT_dict = result_dict
MOT_df = pd.DataFrame.from_dict(MOT_dict, orient='index')

df_age_gender = import_data.df[['ID','NbChild', 'CalculatedIncome','NbTV','NbCellPhones','SocioProfCat','NbCar','NbMoto','NbRoomsHouse']]

df_2 = pd.merge(MOT_df, df_age_gender, left_index=True, right_on="ID")
df_2.loc[df_2['PM'] == 0,'Label'] = 'Public'
df_2.loc[df_2['PM'] == 1,'Label'] = 'Private'
df_2.loc[df_2['PM'] == 2,'Label'] = 'Soft'

df_2.drop(df_2[df_2['NbChild'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['CalculatedIncome'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['NbTV'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['NbCellPhones'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['PM'] < 0].index, inplace = True)

costofchild=700
costoftv=25
costofCP=20
costofcar=800
costofbike=100
costofroom=486

df_2['HighincomeH']= df_2['CalculatedIncome']-(costofcar*df_2['NbCar']+costofbike*df_2['NbMoto']+costofroom*df_2['NbRoomsHouse']+costofchild*df_2['NbChild']+costoftv*df_2['NbTV']+costofCP*df_2['NbCellPhones'])

for index, value in df_2['HighincomeH'].iteritems():
    if value <=1000:
        df_2.loc[index, 'HighincomeH'] = 1
    elif value <=2500:
        df_2.loc[index, 'HighincomeH'] = 2
    elif value <=5000:
        df_2.loc[index, 'HighincomeH'] = 3    
    elif value<=7000:
        df_2.loc[index, 'HighincomeH'] = 4
    elif value <=10000:
        df_2.loc[index, 'HighincomeH'] = 5
    else:
        df_2.loc[index, 'HighincomeH'] = 6


# ------------ STEP 2: data preparation ------------
# Create a filtered dictionary with only PM values of 0, 1, or 2
filtered_dict = {k:v for k,v in result_dict.items() if v['PM'] in [0,1,2]}
# print(filtered_dict)
# print(len(filtered_dict))


MOT_df = pd.DataFrame.from_dict(filtered_dict, orient='index')
MOT_df.loc[MOT_df['PM'] == 0,'Label'] = 'Public'
MOT_df.loc[MOT_df['PM'] == 1,'Label'] = 'Private'
MOT_df.loc[MOT_df['PM'] == 2,'Label'] = 'Soft'


df_extra= import_data.df[['ID', 'age', 'Gender','Income']]
df_final = pd.merge(MOT_df, df_extra, left_index=True, right_on="ID")
df_final.drop(df_final[df_final['Income'] == -1].index, inplace = True)
df_final.drop(df_final[df_final['Gender'] == -1].index, inplace = True)
df_final.drop(df_final[df_final['age'] == -1].index, inplace = True)
df_final.drop(df_final[df_final['PM'] < 0].index, inplace = True)

df_extra_2 = df_2[['ID','HighincomeH','SocioProfCat']]
df_final = pd.merge(df_final, df_extra_2, left_on="ID", right_on="ID")


# ------Plotting Function----------
def acc_plot(k_val,acc,title,filename):
    plt.plot(k_val, acc)
    plt.xlabel('K Nearest Neighbours')
    plt.ylabel('Percentage Accuracy of testing data (%)')
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.show()