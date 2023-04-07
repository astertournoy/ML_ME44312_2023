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

df_age_gender = import_data.df[['ID','NbChild', 'CalculatedIncome','NbTV','NbCellPhones','SocioProfCat']]

df_2 = pd.merge(MOT_df, df_age_gender, left_index=True, right_on="ID")
df_2.loc[df_2['PM'] == 0,'Label'] = 'Public'
df_2.loc[df_2['PM'] == 1,'Label'] = 'Private'
df_2.loc[df_2['PM'] == 2,'Label'] = 'Soft'

df_2.drop(df_2[df_2['NbChild'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['CalculatedIncome'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['NbTV'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['NbCellPhones'] == -1].index, inplace = True)
df_2.drop(df_2[df_2['PM'] < 0].index, inplace = True)

costofchild=300
costoftv=25
costofCP=20
df_2['HighincomeH']= df_2['CalculatedIncome']-(costofchild*df_2['NbChild']+costoftv*df_2['NbTV']+costofCP*df_2['NbCellPhones'])

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

print(df_2)
#Multi-Clustering Code

n_neighbors = 10

#splitting dataset into train, validation and test data
#X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state = 1)

#splitting dataset into train, validation and test data
df_train,df_test = train_test_split(df_2,test_size=0.3,random_state = 1)


# # we only take the first two features. We could avoid this ugly
# # slicing by using a two-dim dataset
# X = df_2[['age','Gender']].to_numpy()
# y = df_2['PM'].to_numpy()

X_train = df_train[['HighincomeH','SocioProfCat']].to_numpy()
Y_train = df_train['PM'].to_numpy()

X_test = df_test[['HighincomeH','SocioProfCat']].to_numpy()
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
        xlabel='Income Classes',
        ylabel='Social Professional Categories',
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
    
    plt.savefig(("q2_train (weights = '%s')" % (weights)), dpi=800)
    plt.show()
    
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)