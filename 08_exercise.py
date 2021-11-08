# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the iris data that is attached here, also the names as column names,
# remove missing values and explore the data a bit.

# width and length of sepals/petals in cm
TITLES = ["sepal length", "sepal width", "petal length", "petal width", "class"]

import pandas as pd
import numpy as np
from sklearn import metrics

df = pd.read_csv('data/iris.data', header=None, names=TITLES)

# replace "NA" with "NaN" for easier dropping
df = df.replace('NA', np.nan, regex=True)
df = df.dropna()

print(df)

### Ex2. Fit a decision tree classifier on the data.
# Response is the class and explanatory (independent) variables are the petal and sepal measurements.
# Here it is not necessary to split the data yet, this is just to give an understanding of decision trees.

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

response_name = "class"

explanatory_names = TITLES
explanatory_names.remove(response_name)

X = df[explanatory_names]   # explanatory
y = df[response_name]       # response

# print(X, y)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X, y)

#Predict the response for test dataset
y_pred = clf.predict(X)

# print(y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy without test-train split:",metrics.accuracy_score(y, y_pred))

### Ex3. Plot the decision tree.
# This should give a rather good insight into how a single classification tree works.
# Hint: the classification tree instance has a plot method that is not listed in its sklearn page.
# See https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree
from sklearn import tree

tree.plot_tree(clf)
#*** I don't see the graphic popping up??? I'm using Linux. ***#

### Ex4. Split the data into training and testing.
# Fit a random forest with suitable parameters to the data.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf2 = RandomForestClassifier(criterion="entropy")
clf2 = clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)

print("Accuracy with RandomForestClassifier test-train split:",metrics.accuracy_score(y_test, y_pred2))

### Ex5. Compare the results to a logistic regression model.
# I.e. fit a logistic regression model on the same training split and compare the prediction results in the test set,
# don't forget to scale the explanatory variables for logistic regression.
# Random forest does not need variable scaling.
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log = log.fit(X_train, y_train)

y_pred3 = log.predict(X_test)
# ConvergenceWarning: STOP: TOTAL NO. of ITERATIONS REACHED LIMIT... Why am I getting this?
print("Accuracy with LogisticRegression test-train split:",metrics.accuracy_score(y_test, y_pred3))

### Ex6. Give a brief explanation of how random forest works.
# You can expect the reader to know how a decision tree works.

answer = '''
A random forest is made up of a large number of individual decision trees that operate as
multiple learning algorithms to obtain better predictive performance than any one algorithm
could obtain alone. Every individual tree produces it's own class prediction and the most
popular class becomes the model's prediction.

The benefit is the large number of uncorrelated models will almost accidentally end up with
the best prediciton.
'''

print(answer)