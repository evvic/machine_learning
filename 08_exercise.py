# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the iris data that is attached here, also the names as column names,
# remove missing values and explore the data a bit.

# width and length of sepals/petals in cm
TITLES = ["sepal length", "sepal width", "petal length", "petal width", "class"]

import pandas as pd
import numpy as np

df = pd.read_csv('data/iris.data', header=None, names=TITLES)

# replace "NA" with "NaN" for easier dropping
df = df.replace('NA', np.nan, regex=True)
df = df.dropna()

print(df)

### Ex2. Fit a decision tree classifier on the data.
# Response is the class and explanatory variables are the petal and sepal measurements.
# Here it is not necessary to split the data yet, this is just to give an understanding of decision trees.