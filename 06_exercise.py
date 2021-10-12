# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Redo Ex6 from the classification
from numpy.random.mtrand import exponential
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

df = pd.read_csv('data/titanic.csv')

# female = 0, male = 1
df['Sex'] = df['Sex'].replace(['female','male'],[0,1])

# 75% for data, 25% for testing
x_train, x_test, y_train, y_test = train_test_split(df.drop(["Survived", "Name"], axis=1), df["Survived"])

LogReg = LogisticRegression(penalty='none').fit(x_train, y_train)

print(x_test)
print(LogReg.predict(x_test))

print(LogReg.score(x_test, y_test))

### Ex2. Plot a bar plot of the logistic regression coefficients of Ex1.
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt

# .coef_ returns the array of coefficients used in log reg model
coefficients = LogReg.coef_[0]
# generate label for each coefficient
labels = list(map(str, list(range(0, len(coefficients)))))

print(LogReg.coef_[0], LogReg.intercept_)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels, coefficients)

# display bar chart
# plt.show()

### Ex3. Redo Ex1 but this time preprocess the explanatory variables with StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df.drop(["Survived", "Name"], axis=1)) # 
somthin = scaler.transform(df.drop(["Survived", "Name"], axis=1))

print(somthin)
x_train, x_test, y_train, y_test = train_test_split(somthin, (df["Survived"]))

LogReg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

LogReg.fit(x_train, y_train)
y_pred = LogReg.predict(x_test)
print("y_pred", y_pred)
score = LogReg.score(x_test , y_test)
print("score: ", score)

### Ex4. Plot a bar plot of the logistic regression coefficients of Ex3. What do these tell you?
print("Ex4 bar plot for logistic regression of ex3...")
# .coef_ returns the array of coefficients used in log reg model
coefficients = LogReg.coef_[0]
# generate label for each coefficient
labels = list(map(str, list(range(0, len(coefficients)))))

print(LogReg.coef_[0], LogReg.intercept_)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels, coefficients)

plt.show()

explanation = ''' 
Comparing the bar plot of coefficients of the original logistic regression model to
the other preprocessed logistic regression model clearly has a difference. The original
model had smaller coefficients overall and were all on average closer to zero. This
tells me that not preprocessing the data into statistal values gives better results for
modeling.
'''

print(explanation)