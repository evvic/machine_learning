# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex 1. Copy and run the code that was shown in the session intro with different parameter values
from operator import length_hint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

# different y_line paramenters
# y_line = 0.5*x + 1
y_line = 0.6*x + 3

# different random value parameters
# y_data = y_line + np.random.randn(100,)/2
y_data = y_line + np.random.randn(100,)/2

df = pd.DataFrame({'x': x, 'y': y_data})

sns.scatterplot(x='x', y='y', data=df)

a, b = np.polyfit(x, y_data, deg=1)
print(a, b)

fig, ax = plt.subplots()
sns.lmplot(x='x', y='y', data=df)
plt.plot(x, y_line, 'r')

### Ex2. Read in the auto-mpg data that we used last time
listy2 = []

# with open (method)
with open('data/auto-mpg.names.txt', 'r') as sumthin:
    for line in sumthin:
        listy2.append(line.strip())

print(listy2)

df = pd.read_csv('data/auto-mpg.data-original.txt', delim_whitespace=True, header=None, names=listy2)
# print(df)

# Ex3. Drop the rows with missing data
# replace "NA" with "NaN" for easier dropping
df = df.replace('NA', np.nan, regex=True)

df = df.dropna()

### Ex4. Split the data into simple training and testing sets. Use 3/4 of data for training and 1/4 for testing
from sklearn.model_selection import train_test_split 

# 75% for data, 25% for testing
df_train, df_test = train_test_split(df, test_size=.25)

### Ex5. Fit a linear regression model mpg ~ weight using the training data
from sklearn.linear_model import LinearRegression

X = df_train["mpg"].values
Y = df_train["weight"].values

X = X.reshape(len(X), 1)
Y = Y.reshape(len(Y), 1)

#                                 (independents, target)
reg_train = LinearRegression().fit(Y, X)

print("r-squared value: ", reg_train.score(Y, X))
# Then predict the mpg for the testing data using the model you just fitted
mpg_pred = reg_train.predict(Y)
# print("predicted mpg: ", mpg_pred)

# Calculate the mean squared error for the prediction. Is this a good fit?

### Ex6. Fit a linear regression model mpg ~ weight + horsepower
X2 = df_train["mpg"].values.reshape(len(X), 1)
Y2 = df_train[["weight", "horsepower"]]

#                                  (independents, target)
reg2_train = LinearRegression().fit(Y2, X2)

# Calculate the mean squared error for the prediction
from sklearn.metrics import mean_squared_error

mpg_pred = reg2_train.predict(Y2)

rsq_err = mean_squared_error(X2, mpg_pred)

print("r-squared error prediction: ", rsq_err)

# Did the results improve by adding horsepower as an explanatory variable?
rsq_diff = reg_train.score(Y, X) - reg2_train.score(Y2, X2)
print("r_squared difference: ", rsq_diff)
if rsq_diff > 0:
    # then adding horsepower did not improve results
    print("adding horsepower as an explanatory variable did NOT improve results.\n")
else:
    # then adding horsepower DID improve results
    print("adding horsepower as an explanatory variable DID improve results.\n")


### Ex7. Try even more variables to explain the mpg. 
X3 = df_train["mpg"].values.reshape(len(X), 1)
Y3 = df_train[["weight", "horsepower", "acceleration", "cylinders"]]

#                                  (independents, target)
reg3_train = LinearRegression().fit(Y3, X3)

# Do the prediction results always improve when adding more explanatory variables?
rsq_diff2 = reg_train.score(Y, X) - reg3_train.score(Y3, X3)
print("r_squared difference between w/ horsepower and even more variables: ", rsq_diff2)
if rsq_diff2 > 0:
    # then adding even more variables did not improve results
    print("adding even more explanatory variables did NOT improve results.\n")
else:
    # then adding even more variables DID improve results
    print("adding even more explanatory variables DID improve results.\n")

print("Prediction results don't always improve by adding more explanatory variables such as 'origin' and 'car name' wouldn't improve results")



