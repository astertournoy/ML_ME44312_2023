# #import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# import import_data
# df = import_data.df
# # df.head()
# # df.describe()

# #implement a scatter plot for the data (see example in the slides)
# # plt.scatter(df.ID, df.DestAct)
# # plt.show()


# # sort out choices of transport mode per ID
# df.sort_index()




# # We make a dictionary per ID, that says how often that ID chooses 0 (public transport),\
# # 1 (private mode) or 2 (soft mode)


# # Create an empty dictionary to store the results
# result_dict = {}
# # Loop through each unique index value
# for idx in df['ID'].unique():
#     # Subset the dataframe for the current index value
#     sub_df = df[df['ID'] == idx]
#     # Count the occurrences of 0, 1, and 2 in the 'Choice' column
#     count_0 = (sub_df['Choice'] == 0).sum()
#     count_1 = (sub_df['Choice'] == 1).sum()
#     count_2 = (sub_df['Choice'] == 2).sum()
#     # Add the counts to the dictionary
#     result_dict[idx] = {'public_transport': count_0, 'private_mode': count_1, 'soft_mode': count_2}

# # print(len(result_dict))


# # Create an empty list to store the public_transport values
# public_transport_values = []
# private_mode_values = []
# soft_mode_values = []

# # add values of data to lists
# for id, values in result_dict.items():
#     public_transport_values.append(values['public_transport'])
#     private_mode_values.append(values['private_mode'])
#     soft_mode_values.append(values['soft_mode'])


# # plot data
# x = public_transport_values
# y = private_mode_values
# z = soft_mode_values

# #
# modes_of_transport = {
#     1: "Private modes",
#     2: "Public modes",
#     3: "Soft modes"
# }

# # Create the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)

# # Label the axes
# ax.set_xlabel('Public Transport')
# ax.set_ylabel('Private Mode')
# ax.set_zlabel('Soft Mode')

# # Show the plot
# plt.show()

# x2 = df[['NbChild', 'Income', 'NbTV', 'Education','NbCellPhones']]
# y2 = df['transport_mode']

# # Split the data into training and testing sets
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3)

# # Create a decision tree classifier
# classifier = DecisionTreeClassifier()

# # Fit the classifier to the training data
# classifier.fit(x2_train, y2_train)

# # Make predictions on the testing data
# y2_pred = classifier.predict(x2_test)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y2_test, y2_pred)
# print("Accuracy:", accuracy)
#Libraries and Modules
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

#Datasets
import Subquestion_1
import import_data

#Cleaning and creating final dataframe
MOT_dict = Subquestion_1.result_dict
MOT_df = pd.DataFrame.from_dict(MOT_dict, orient='index')

df_age_gender = import_data.df[['ID','age', 'Gender']]

df_4 = pd.merge(MOT_df, df_age_gender, left_index=True, right_on="ID")
df_4.loc[df_4['PM'] == 0,'Label'] = 'Public'
df_4.loc[df_4['PM'] == 1,'Label'] = 'Private'
df_4.loc[df_4['PM'] == 2,'Label'] = 'Soft'

df_4.drop(df_4[df_4['NbChild'] == -1].index, inplace = True)
df_4.drop(df_4[df_4['Income'] == -1].index, inplace = True)
df_4.drop(df_4[df_4['NbTV'] == -1].index, inplace = True)
df_4.drop(df_4[df_4['NbCellPhones'] == -1].index, inplace = True)
df_4.drop(df_4[df_4['PM'] < 0].index, inplace = True)


#Multi-Clustering Code

n_neighbors = 10

#splitting dataset into train, validation and test data
#X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state = 1)

#splitting dataset into train, validation and test data
df_train,df_test = train_test_split(df_4,test_size=0.3,random_state = 1)


# # we only take the first two features. We could avoid this ugly
# # slicing by using a two-dim dataset
# X = df_4[['age','Gender']].to_numpy()
# y = df_4['PM'].to_numpy()

X_train = df_train[['NbChild','Income', 'NbTV', 'NbCellPhones']].to_numpy()
Y_train = df_train['PM'].to_numpy()

X_test = df_test[['NbChild','Income', 'NbTV', 'NbCellPhones']].to_numpy()
Y_test = df_test['PM'].to_numpy()

# Create color maps
cmap_light = ListedColormap(["grey", "green", "lightgrey"])
cmap_bold = ["green", "orange", "darkblue"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, Y_train)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel='Age',
        ylabel='Gender',
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=X_train[:, 0],
        y=X_train[:, 1],
        hue=df_train['Label'].to_numpy(),
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "3-Class training classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    
    plt.savefig(("q4_train (weights = '%s')" % (weights)), dpi=800)
    plt.show()
    
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

