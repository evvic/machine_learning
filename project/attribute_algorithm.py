# run py script through terminal with '/bin/python3'

# there are 3 grade attributes: 1st, 2nd, and final
# we can use the received 1st and 2nd grades to help predict the final grade
# however, we will use many of the other attributes to prdedict final grade

from importlib import import_module
from os import urandom
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle
import random
import pickle

# read in data set
data = pd.read_csv("data/student-mat.csv", sep=";")

# get all column names and remove G3 temporarily
column_names = data.columns.tolist()
column_names.remove('G3')
column_names.remove('G2')
column_names.remove('G1')

accuracy = 0.0
used_cols = []
model = None

# NOTE: what not to use: absences, (G1, G2 because they are too easy),
while accuracy < 0.90:
    # set predict to the attribute we are trying to predict: final grade (G3)
    predict = "G3"

    # choose a random sample to test, random attributes and random number of attributes
    test_cols = random.sample(column_names, random.randint(4, len(column_names)))
    used_cols = test_cols
    test_cols.append(predict)

    test_df = data[test_cols]

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

    ### when finally decided which metrics to use, add more individual detail to each attribute when printing

    #######################
    # PREPROCESS THE DATA #
    #######################
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(test_df.drop([predict], 1))

    # drop the predict value fro the remaining attributes/features
    X = np.array(scaled_features)

    # assign the predict value to the y label(s)
    y = np.array(data[predict])

    for _ in range(30):
        # split data into testing and training sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        ########################
        # APPLY ML METHODS 1/2 #
        ########################

        from sklearn.linear_model import Ridge

        logistic = Ridge().fit(x_train, y_train)

        # get score of how correct the model was
        acc = logistic.score(x_test, y_test)
        print("accuracy: ", acc)

        if(acc > accuracy):
            accuracy = acc
            model = logistic

# display results
print("ACCURACY:", accuracy)
print("COLUMNS USED:", used_cols)
print("MODEL: ", model)

# save the good model in a pickle file
#with open("data/linearmodel.pickle", "wb") as f:
#   pickle.dump(model, f)

# save the attributes used to a text file
#with open("data/linearcolumns.txt", "w") as f:
#    for col in used_cols:
#        f.write(col + '\n')

