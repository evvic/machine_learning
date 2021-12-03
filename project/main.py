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

# filter to only desired attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# set predict to the attribute we are trying to predict: final grade (G3)
predict = "G3"

# drop the predict value fro the remaining attributes/features
X = np.array(data.drop([predict], 1))

# assign the predict value to the y label(s)
y = np.array(data[predict])

# split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


linear = linear_model.LinearRegression()

# fitting the linear model to the training data
linear.fit(x_train, y_train)

###################
# DISPLAY RESULTS #
###################
import matplotlib.pyplot as pyplot
from matplotlib import style

# style scatterplot
style.use("ggplot")


pyplot.scatter(data["G1"], data[predict])
pyplot.scatter(data["G2"], data[predict])
pyplot.scatter(data["absences"], data[predict])
pyplot.xlabel("G1")
pyplot.ylabel("final grade")

pyplot.show()