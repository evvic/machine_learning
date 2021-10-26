# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the auto-mpg data and drop the rows with missing data.
from os import listxattr
from typing import List
import pandas as pd
listy2 = []

# with open (method)
with open('data/auto-mpg.names.txt', 'r') as sumthin:
    for line in sumthin:
        listy2.append(line.strip())

print(listy2)

df = pd.read_csv('data/auto-mpg.data-original.txt', delim_whitespace=True, header=None, names=listy2)

df = df.dropna()

print(df.head())
df = df.drop(columns=['car name'])


### Ex2. Split the data into simple training and testing sets. Use 3/4 of data for training and 1/4 for testing.
from sklearn.model_selection import train_test_split 

# Use fixed random_state parameter.
df_train, df_test = train_test_split(df, random_state=3)

### Ex3. Fit a standard scaler from sklearn.preprocessing to the training set.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df_train)

### Ex4. Fit a linear regression model into the data with mpg as the response and all other columns except car name as the explanatory variables
from sklearn.linear_model import LinearRegression

listy2.remove("car name")
listy2.remove("mpg")
col_names = listy2

x_train, x_test, y_train, y_test = train_test_split(df[col_names], df["mpg"], random_state=3)

#                                 (independents, target)
reg_train = LinearRegression().fit(x_train, y_train)

### Ex5. Predict mpg on the testing set, remember to also apply the standard scaler to the testing set now
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

y_pred = reg_train.predict(x_test)

test_scaler = StandardScaler() #.fit(x_test)

plt.subplot(122)
p = reg_train.predict(test_scaler.fit_transform(x_test))
sns.kdeplot(p)
plt.title('Predict MPG with Standard Scaler')

plt.show()

#  Calculate some evaluation metrics and plot the coefficients
print("r-squared error prediction: ", mean_squared_error(y_test, y_pred))
print("score: ", reg_train.score(x_test , y_test))

coefficients = reg_train.coef_

print("Coefficients of Linear Regression line: ", coefficients)
print("The y-intercept of the Linear Reression Line", reg_train.intercept_)

labels = list(map(str, list(range(0, len(coefficients)))))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels, coefficients)
plt.show()

### Ex6. Redo parts 4 and 5 with ridge regression and lasso regression. 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# function to reuse code from parts 4 & 5 for using ridge and lasso regression
def RegMetrics(reg_train, linear_model):

    x_train, x_test, y_train, y_test = train_test_split(df[col_names], df["mpg"], random_state=3)

    y_pred = reg_train.predict(x_test)

    test_scaler = StandardScaler() #.fit(x_test)

    plt.subplot(122)
    p = reg_train.predict(test_scaler.fit_transform(x_test))
    sns.kdeplot(p)
    plt.title('Predict MPG with ' + linear_model)

    plt.show()

    #  Calculate some evaluation metrics and plot the coefficients
    print("r-squared error prediction: ", mean_squared_error(y_test, y_pred))
    print("score: ", reg_train.score(x_test , y_test))

    coefficients = reg_train.coef_

    print("Coefficients of Linear Regression line: ", coefficients)
    print("The y-intercept of the Linear Reression Line", reg_train.intercept_)

    labels = list(map(str, list(range(0, len(coefficients)))))

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels, coefficients)
    plt.show()


# Ridge Regression metrics and coefficients
RegMetrics(Ridge().fit(x_train, y_train), linear_model="Ridge")

# Lasso Regression metrics and coefficients
RegMetrics(Lasso().fit(x_train, y_train), linear_model="Lasso")

# You should see differences in the coefficients between the different models. 
# Most notably some of the coefficients in lasso regression are 0. 
# Why do some of the coefficients shrink to zero in lasso case?

answer = '''
The Lasso Regressions's coefficients were defintitely different from Ridge or Linear. 
The first 4 coefficients were virtually zero but the last coefficient was very large.
Lasso performs shrinkage creating corners in the constraint, making a diamond shape in 2D,
then if the sum of squares lands in a corner then the coeffiecient axis minimizes to zero.

Both Linear and Ridge Regression coefficients appeared very similar.
'''
print(answer)

# P.S. Thanks for telling me you appreciate being able to easily read my code! 