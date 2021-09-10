# run py script through terminal with '/bin/python3'
# Eric Brown

## Ex 0. Install Seaborn
# sudo pip3 install seaborn

## Ex 1. with open, read, split
# auto-mpg.names.txt
listy2 = []

# with open (method)
with open('data/auto-mpg.names.txt', 'r') as sumthin:
    for line in sumthin:
        listy2.append(line.strip())

print(listy2)

## Ex 2. pandas read_csv, dataframe, head
import pandas as pd

df = pd.read_csv('data/auto-mpg.data-original.txt', delim_whitespace=True, header=None, names=listy2)

#print first 5 with head()
print(df.head())

## Ex 3. info, describe
print("df.info()")
print(df.info())
print("df.descibe()")
print(df.describe())

## Ex 4. isna, numpy any
import numpy

# NA values return True mapped in the DF
print("df.isna()")
print(df.isna())

# Test whether any array element along a given axis evaluates to True
print("numpy.any(df)")
print(numpy.any(df))

## Ex 5. seaborn pairplot
import seaborn
import matplotlib.pyplot as plt

# create pair plot comparing weight to MPG
seaborn.pairplot(df, x_vars="weight", y_vars="mpg", kind='scatter')

# display graph in separate window using matplotlib
# plt.show()

## Ex 6. seaborn lmplot
seaborn.lmplot(x="weight", y="mpg", data=df)
# the line is the best fit for linear correlation between the 2 sets of data when being compared
# the shaded area around the regression line is the confidence interval of the regression estimates

plt.show()