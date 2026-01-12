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
clf = tree.DecisionTreeClassifier(max_depth=3)

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

