### sources:
# https://www.youtube.com/watch?v=BOhgGA7Eu5E&ab_channel=TechWithTim
# https://www.youtube.com/watch?v=45ryDIPHdGg&ab_channel=TechWithTim

# run py script through terminal with '/bin/python3'

# data is mulitavariative: more than 2 variables
# number of instances (students): 649
# number of attributes (per instance): 33

# there are 3 grade attributes: 1st, 2nd, and final
# we can use the received 1st and 2nd grades to help predict the final grade
# however, we will use many of the other attributes to prdedict final grade

from importlib import import_module
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle

# read in data set
data = pd.read_csv("data/student-mat.csv", sep=";")

####################
# EXPLORE THE DATA #
####################

#ex04

# NOTE: what not to use: absences, (G1, G2 because they are too easy),

# filter to only desired attributes
data = data[["G3", "G1", "G2", "activities", "paid", "studytime", "Dalc", "Fedu", "Medu",  "famsup", "higher",
    "Pstatus" ]]

data['activities'] = data['activities'].map({'yes': 1, 'no': 0})
data['paid'] = data['paid'].map({'yes': 1, 'no': 0})
#data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
#data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})
# set predict to the attribute we are trying to predict: final grade (G3)
predict = "G3"

# describe the final grade
print("FINAL GRADES STATISTICS (", predict, ")\n", data[predict].describe(), '\n')

### when finally decided which metrics to use, add more individual detail to each attribute when printing

# print metrics of columns/attributes
for col in data.columns:
    if(col != predict):
        print(col.upper(), 'STATISTICS\n', data[col].describe(), '\n')


#######################
# PREPROCESS THE DATA #
#######################

# drop the predict value fro the remaining attributes/features
X = np.array(data.drop([predict], 1))

# assign the predict value to the y label(s)
y = np.array(data[predict])

# split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

########################
# APPLY ML METHODS 1/2 #
########################

linear = linear_model.LinearRegression()

# fitting the linear model to the training data
linear.fit(x_train, y_train)

# array of predictiosn for each student given the attributes
predictions = linear.predict(x_test)

# print each individual prediction with true value and attributes
for x in range(len(predictions)):
    if(predictions[x].round() == y_test[x]):
        print(predictions[x].round(), '\t', y_test[x], "\tCORRECT!", x_test[x])
    else:
        print(predictions[x].round(), '\t', y_test[x], "\tX   \t", x_test[x])

# get score of how correct the model was
acc = linear.score(x_test, y_test)
print("accuracy: ", acc)

########################
# APPLY ML METHODS 2/2 #
########################

# use a second method here????

###################
# DISPLAY RESULTS #
###################
import matplotlib.pyplot as pyplot
from matplotlib import style

# style scatterplot
style.use("ggplot")

# print metrics of columns/attributes
for col in data.columns:
    if(col != predict):
        pyplot.scatter(data[col], data[predict], label=col)

#pyplot.scatter(data["G1"], data[predict], label="G1")
#pyplot.scatter(data["G2"], data[predict], label="G2")
#pyplot.scatter(data["absences"], data[predict])
# pyplot.scatter(data["failures"], data[predict])
pyplot.xlabel("Attributes")
pyplot.ylabel("final grade")
pyplot.legend(loc="upper left")

pyplot.show()