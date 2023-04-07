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
import Data_Prep

df_4 = Data_Prep.df_final

df_train = Data_Prep.df_final

#Multi-Clustering Code
n_neighbors = 15


#splitting dataset into train, validation and test data
#X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state = 1)

#splitting dataset into train, validation and test data

df_train,df_test = train_test_split(df_4,test_size=0.3,random_state = 1)


df_train,df_test = train_test_split(df_train,test_size=0.3,random_state = 1)


X_train = df_train[['age','Gender']].to_numpy()
Y_train = df_train['PM'].to_numpy()

X_test = df_test[['age','Gender']].to_numpy()
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