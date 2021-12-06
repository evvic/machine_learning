
from importlib import import_module
from os import urandom
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle
import random
import pickle

def convertToBin(test_df):
    ### convert all potential non-int or non-binary values
    if "school" in test_df.columns:
        test_df['school'] = test_df['school'].map({'GP': 1, 'MS': 0})
    if 'sex' in test_df.columns:
        test_df['sex'] = test_df['sex'].map({'F': 1, 'M': 0})
    if 'address' in test_df.columns:
        test_df['address'] = test_df['address'].map({'U': 1, 'R': 0})
    if 'famsize' in test_df.columns:
        test_df['famsize'] = test_df['famsize'].map({'LE3': 1, 'GT3': 0})
    if 'Pstatus' in test_df.columns:
        test_df['Pstatus'] = test_df['Pstatus'].map({'T': 1, 'A': 0})
    if 'Mjob' in test_df.columns:
        test_df['Mjob'] = test_df['Mjob'].map({'teacher': 5, 'health': 4, 'services': 3, 'at_home': 2, 'other': 1})
    if 'Fjob' in test_df.columns:
        test_df['Fjob'] = test_df['Fjob'].map({'teacher': 5, 'health': 4, 'services': 3, 'at_home': 2, 'other': 1})
    if 'reason' in test_df.columns:
        test_df['reason'] = test_df['reason'].map({'reputation': 4, 'course': 3, 'home': 2, 'other': 1})
    if 'guardian' in test_df.columns:
        test_df['guardian'] = test_df['guardian'].map({'mother': 3, 'father': 2, 'other': 1})
    if 'schoolsup' in test_df.columns:
        test_df['schoolsup'] = test_df['schoolsup'].map({'yes': 1, 'no': 0})
    if 'famsup' in test_df.columns:
        test_df['famsup'] = test_df['famsup'].map({'yes': 1, 'no': 0})
    if 'paid' in test_df.columns:
        test_df['paid'] = test_df['paid'].map({'yes': 1, 'no': 0})
    if 'activities' in test_df.columns:
        test_df['activities'] = test_df['activities'].map({'yes': 1, 'no': 0})
    if 'nursery' in test_df.columns:
        test_df['nursery'] = test_df['nursery'].map({'yes': 1, 'no': 0})
    if 'higher' in test_df.columns:
        test_df['higher'] = test_df['higher'].map({'yes': 1, 'no': 0})
    if 'internet' in test_df.columns:
        test_df['internet'] = test_df['internet'].map({'yes': 1, 'no': 0})
    if 'romantic' in test_df.columns:
        test_df['romantic'] = test_df['romantic'].map({'yes': 1, 'no': 0})

    return test_df

# read in data set
data = pd.read_csv("data/student-mat.csv", sep=";")

column_names = []
predict = 'G3'

with open("data/linearcolumns.txt", "r") as f:
    column_names = f.readlines()
    # remove newlines from each string
    column_names = list(map(lambda x: x.replace('\n',''), column_names))

print(column_names)

accuracy = 0.0

# to open the pickle file
pickle_in = open("data/linearmodel.pickle", "rb")
model = pickle.load(pickle_in)

data = convertToBin(data[column_names])

print(data)

# drop the predict value from the remaining attributes/features
X = np.array(data.drop([predict], 1))

# assign the predict value to the y label(s)
y = np.array(data[predict])

# split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#don't have to refit the model

# get score of how correct the model was
acc = model.score(x_test, y_test)

print("accuracy: ", acc)




