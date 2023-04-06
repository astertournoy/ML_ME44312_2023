import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.cluster import KMeans

import Subquestion_1
import import_data

#Cleaning and creating final dataframe
MOT_dict = Subquestion_1.result_dict
MOT_df = pd.DataFrame.from_dict(MOT_dict, orient='index')

# #Creating wealth index 
# df_wealth_index = import_data.df[['ID', 'NbChild', 'Income', 'NbTV','NbCellPhones']]

# df_wealth_index.drop(df_wealth_index[df_wealth_index['NbChild'] == -1].index, inplace = True)
# df_wealth_index.drop(df_wealth_index[df_wealth_index['Income'] == -1].index, inplace = True)
# df_wealth_index.drop(df_wealth_index[df_wealth_index['NbTV'] == -1].index, inplace = True)
# df_wealth_index.drop(df_wealth_index[df_wealth_index['NbCellPhones'] == -1].index, inplace = True)

# # perform k-means clustering on the dataframe
# kmeans = KMeans(n_clusters=5, random_state=0)
# kmeans.fit(df_wealth_index)

# # get the cluster assignments for each data point
# df_wealth_index['cluster'] = kmeans.predict(df_wealth_index)

# # print the resulting dataframe with the cluster index
# #print(df_wealth_index)

# df_profession = import_data.df[['ID', 'SocioProfCat']]

df_income_profession = import_data.df[['ID', 'Income', 'SocioProfCat']]

#df_merge = pd.merge(MOT_df, df_wealth_index, df_profession, left_index=True, right_on="ID")
df_merge = pd.merge(MOT_df, df_income_profession, left_index=True, right_on="ID")
df_merge.loc[df_merge['PM'] == 0,'Label'] = 'Public'
df_merge.loc[df_merge['PM'] == 1,'Label'] = 'Private'
df_merge.loc[df_merge['PM'] == 2,'Label'] = 'Soft'

df_merge.drop(df_merge[df_merge['Income'] == -1].index, inplace = True)
df_merge.drop(df_merge[df_merge['SocioProfCat'] == -1].index, inplace = True)
df_merge.drop(df_merge[df_merge['PM'] < 0].index, inplace = True)


#Multi-Clustering Code
n_neighbors = 6

#splitting dataset into train, validation and test data
df_train,df_test = train_test_split(df_merge,test_size=0.3,random_state = 1)


X_train = df_train[['Income', 'SocioProfCat']].to_numpy()
Y_train = df_train['PM'].to_numpy()

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
        xlabel='Income',
        ylabel='SocioProfCat',
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
    
    # plt.savefig(("q4_train (weights = '%s')" % (weights)), dpi=800)
    plt.show()


