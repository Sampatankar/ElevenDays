"""
Cross-Validation chapter 2 (pg. 14) from Thakur.

The winequality-red.csv file from this link: https://www.kaggle.com/datasets/anairamcosta/winequality-red-csv

To work through the use of cross-validation to counter overfitting we use this dataset.  According to the paper, Cortex et al., 2009 they gave red wine 11 attributes:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulphur dioxide
7. total sulphur dioxide
8. density
9. pH
10. sulphates
11. alcohol


"""

import pandas as pd

df = pd.read_csv("winequality-red.csv")

# print(df)

# mapping dictionary that maps the quality values from 0 to 5:
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}

# You can use the map function of pandas with any dictionary to convert the values in a given column to values in the dictionary
df.loc[:, "quality"] = df.quality.map(quality_mapping)


# Splitting the dataset into traning and test/dev sets:
# Use sample with frac=1 to shuffle the dataframe. We reset the indices since they change after shuffling the dataframe:
df = df.sample(frac=1).reset_index(drop=True)

# Top 1000 rows are selected for training:
df_train = df.head(1000)

# Top 599 values are selected for test/dev:
df_test = df.tail(599)

# print(df_train)
# print(df_test)

# TRAINING A DECISION-TREE MODEL ON THE TRAINING SET
# For the decision-tree we use sci-kit learn:
# import from sci-kit learn:
from sklearn import tree
from sklearn import metrics

# initialise decision tree classifier class with a max-depth of 3:
# clf = tree.DecisionTreeClassifier(max_depth=3)

# Modified to max-depth of 7:
clf = tree.DecisionTreeClassifier(max_depth=7)

# Choose the columns you want to train on, which will be the features of the model:
cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# train the model on the provided features and mapped quality from before:
clf.fit(df_train[cols], df_train.quality)


# TESTING THE ACCURACY OF THE MODEL ON THE TRAINING SET AND THE TEST SET:
# generate predictions on the training set:
train_predictions = clf.predict(df_train[cols])

# generate predictions on the test set:
test_predictions = clf.predict(df_test[cols])

# calculate the accuracy of predictions on training data set:
train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)

# calculate the accuracy of predictions on test data set:
test_accuracy = metrics.accuracy_score(df_test.quality, test_predictions)


# print(f"Training predictions: ", train_predictions)
# print(f"Test predictions: ", test_predictions)
# print(f"Train accuracy: ", train_accuracy)
# print(f"Test accuracy: ", test_accuracy)

"""
==========================================================================================================
1. Comment out the print statements above, and now you can run the section below

"""


# CALCULATE ACCURACIES FOR DIFFERENT VALUES OF OF MAX_DEPTH AND MAKING A PLOT:
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# this is our global size of label text on the plots
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

#ensure plot is displayed inside the notebook (for use in Jupyter notebooks)
# %matplotlib inline

# initialise lists to store accuracies for training and test data - starting with 50% accuracy:
train_accuracies = [0.5]
test_accuracies = [0.5]

# iterate over a few depth values:
for depth in range(1,25):
    # initialise the model:
    clf = tree.DecisionTreeClassifier(max_depth=depth)

    # columns/features for training note that, this can be done outside the loop
    # has been initialised already above, but could be placed at this level of indent

    # fit the model on given features:
    clf.fit(df_train[cols], df_train.quality)

    # create training & test predictions:
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])

    # calculate training and test accuracies:
    train_accuracy = metrics.accuracy_score(
        df_train.quality, train_predictions
    )
    test_accuracy = metrics.accuracy_score(
        df_test.quality, test_predictions
    )

    # append accuracies:
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# create two plots using matplotlib and seaborn:
# plt.figure(figsize=(10, 5))
# sns.set_style("whitegrid")
# plt.plot(train_accuracies, label="train accuracy")
# plt.plot(test_accuracies, label="test accuracy")
# plt.legend(loc="upper left", prop={'size': 15})
# plt.xticks(range(0, 26, 5))
# plt.xlabel("max depth", size=20)
# plt.ylabel("accuracy", size=20)
# plt.show()

"""
=================================================================================
2. Uncomment the create two plots using matplotlib and seaborn section to run it, then comment out again to run the next section.
"""

from sklearn import model_selection

"""
splitting data into k-equal parts using K-fold from sci-kit learn
"""

# if __name__ == "__main__":
#     # Training data is in a CSV file:
#     df = pd.read_csv("train.csv")

#     # we create a mew column called kfold and fill it with -1:
#     df["kfold"] = -1

#     # the next step is to randomise the rows of the data:
#     df = df.sample(frac=1).reset_index(drop=True)

#     # initialise the kfold class from the model_selection module:
#     kf = model_selection.KFold(n_splits=5)

#     # fill the new kfold column:
#     for fold, (trn_, val_) in enumerate(kf.split(X=df)):
#         df.loc[val_, 'kfold'] = fold

#     # save the new csv with a kfold column:
#     df.to_csv("train_folds.csv", index=False)


"""
splitting the data into k-equal parts using stratified k-fold cross-validation from sklearn
"""

# if __name__ == "__main__":
#     # Training data is in a CSV file:
#     df = pd.read_csv("train.csv")

#     # we create a mew column called kfold and fill it with -1:
#     df["kfold"] = -1

#     # the next step is to randomise the rows of the data:
#     df = df.sample(frac=1).reset_index(drop=True)

#     # fetch targets:
#     y = df.target.values

#     # initialise the kfold class from the model_selection module:
#     kf = model_selection.StratifiedKFold(n_splits=5)

#     # fill the new stratified kfold column:
#     for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
#         df.loc[v_, 'strat-kfold'] = f

#     # save the new csv with a kfold column:
#     df.to_csv("train_folds.csv", index=False)



"""
===================================================================================================
3. modify the code above when you have you own dataset to use.
Below is the use of k-fold cross-validation for a regression problem, rather than the classification problems above.
"""

# stratified k-fold for regression:
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    # we create a new column called kfold and fill it with -1
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule (taking the floor value but also you can round it):
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets:
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module:
    kf = model_selection.StratifiedKFold(n_splits=4)

    # fill the new kfold column - note we now use bins, not targets:
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins columns:
    data = data.drop("bins", axis=1)
    # return dataframe with folds:
    return data


if __name__ == "__main__":
    # we create a sample dataset with 15000 samples and 100 features and 1 target:
    X, y = datasets.make_regression(
        n_samples = 15000, n_features = 100, n_targets = 1
    )

    # create a dataframe out of our numpy arrays:
    df = pd.DataFrame(
        X,
        columns = [f"f_{i}" for i in range(X.shape[1])]
    )
    df.loc[:, "target"] = y

    # create folds:
    df = create_folds(df)

