#Libraries and Modules
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import pandas as pd

#Datasets
import Data_Prep

#Cleaning and creating final dataframe
# MOT_dict = Subquestion_1.result_dict
# MOT_df = pd.DataFrame.from_dict(MOT_dict, orient='index')

# df_age_gender = import_data.df[['ID','age', 'Gender']]

# df_4 = pd.merge(MOT_df, df_age_gender, left_index=True, right_on="ID")
# df_4.loc[df_4['PM'] == 0,'Label'] = 'Public'
# df_4.loc[df_4['PM'] == 1,'Label'] = 'Private'
# df_4.loc[df_4['PM'] == 2,'Label'] = 'Soft'

# df_4.drop(df_4[df_4['Gender'] == -1].index, inplace = True)
# df_4.drop(df_4[df_4['age'] == -1].index, inplace = True)
# df_4.drop(df_4[df_4['PM'] < 0].index, inplace = True)


df_train = Data_Prep.df_final

#Multi-Clustering Code

n_neighbors = 15

#splitting dataset into train, validation and test data
#X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state = 1)

#splitting dataset into train, validation and test data
df_train,df_test = train_test_split(df_train,test_size=0.3,random_state = 1)

# # we only take the first two features. We could avoid this ugly
# # slicing by using a two-dim dataset
# X = df_4[['age','Gender']].to_numpy()
# y = df_4['PM'].to_numpy()

X_train = df_train[['age','Gender']].to_numpy()
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
    
