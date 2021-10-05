# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the winequality data attached to this exercise
import pandas as pd

df = pd.read_csv('data/winequality-red.csv', sep=';')

### Ex2. Explore the data: remove rows with missing values
df = df.dropna()

# print metrics of columns
print("number of columns:" + str(len(df.columns)))

for col in df.columns:
    print(col, ": ", df[col].describe())

# draw at least one figure and write a caption for it that interprets the figure.
import seaborn
import matplotlib.pyplot as plt

# create pair plot comparing weight to MPG
seaborn.pairplot(df, x_vars="alcohol", y_vars="quality", kind='reg')
# plt.show()

'''
While using just the alcohol and quality traits of the wine data does provide a regression,
I think other factors are needed because the line of best fit between these 2 values has a 
lot of error. However, the trend does suggest that with more alcohol, the quality rises.
'''

### Ex3. Write a function that:
# fits a ridge regression model on the data to predict the 'quality' column using 10-fold cross-validation.
# returns the mean absolute errors and coefficient of determinations (r2) of each fold, 
# i.e., two lists of 10 values: other list contains the maes and the other the r2s.
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score
from numpy import absolute, std, mean

def RidgeRegError(alpha=1.0):
    
    X, y = df['quality'].values.reshape(-1, 1), df.values[:, -1]

    #X.info
    print('X info')
    print(type(X))
    print(X)

    print('y info')
    print(type(y))
    print(y)

    # define model
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    print(scores)

    r2scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

    print(r2scores)

    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

    return [scores, r2scores]

ret = RidgeRegError()

print("returned vals")
print(ret)

### Ex4. Test different values of alpha for the ridge regression model
# Draw boxplots of the maes and r2s for three different values of alpha: 1, 10, 100.


