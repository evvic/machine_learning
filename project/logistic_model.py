
from importlib import import_module
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle

# read in data set
data = pd.read_csv("data/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3"]]

# set predict to the attribute we are trying to predict: final grade (G3)
predict = "G3"

#######################
# PREPROCESS THE DATA #
#######################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#scaled_features = scaler.fit_transform(data.drop([predict], 1))

# drop the predict value fro the remaining attributes/features
X = np.array(data.drop([predict], 1))

# assign the predict value to the y label(s)
y = np.array(data[predict])

# split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

########################
# APPLY ML METHODS 2/2 #
########################
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

#model = LogisticRegression(solver="lbfgs", max_iter=100000).fit(x_train, y_train)
model = Ridge(alpha=0.1).fit(x_train, y_train)

# get score of how correct the logistic model was
acc = model.score(x_test, y_test)
print("logistic accuracy: ", acc)

# use a second method here????