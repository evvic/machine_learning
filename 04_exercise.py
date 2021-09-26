# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the titanic data attached to this exercise
import pandas as pd

df = pd.read_csv('data/titanic.csv')

print(df)

### Ex2. Explore the data

# remove rows with missing values
df = df.dropna()
print(df)

# print metrics of columns
for col in df.columns:
    if(col == "Name" or col == "Sex"):
        print("can't perform metrics on name or sex")
    else:
        print(col, ": ", df[col].describe())


# draw at least one figure and write a caption for it that interprets the figure.
import seaborn
import matplotlib.pyplot as plt

# create pair plot comparing weight to MPG
seaborn.pairplot(df, x_vars="Survived", y_vars="Age", kind='scatter')

seaborn.lmplot(x="Survived", y="Age", data=df)
# the line is the best fit for linear correlation between the 2 sets of data when being compared
# the shaded area around the regression line is the confidence interval of the regression estimates

# plt.show()
# this comparaes age to surviving 

### Ex3. Split the data into train and test datasets with 20% of the data for testing
from sklearn.model_selection import train_test_split 

# 75% for data, 25% for testing
df_train, df_test = train_test_split(df, test_size=.20)

### Ex4. Train a logistic regression model with the training dataset using 'Pclass' as the response and 'Fare' as the explanatory variable
from sklearn.linear_model import LogisticRegression

X = df_train["Pclass"].values
Y = df_train["Fare"].values

X = X.reshape(len(X), 1)
Y = Y.reshape(len(Y), 1)

#                                   (independents, target)
reg_train = LogisticRegression(penalty='none').fit(Y, X)

print("r-squared value: ", reg_train.score(Y, X))

pred = reg_train.predict(Y)

print(pred)
print(Y)

from sklearn.metrics import mean_squared_error

rsq_err = mean_squared_error(X, pred)

print("r-squared error prediction: ", rsq_err)

### Ex5. Find out and explain (like i'm five) what the following performance metrics mean
# precision
from sklearn.metrics import precision_score
print('precision score: ', precision_score(X, pred, average='micro'))
# the ratio of correctly predicted positive observations to the total predicted positive observations

# recall
from sklearn.metrics import recall_score
print('recall score: ', recall_score(X, pred, average='micro'))

# f-score.
