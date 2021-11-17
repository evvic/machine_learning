# run py script through terminal with '/bin/python3'
# Eric Brown

### Ex1. Read in the iris data that is attached here, remove missing values and explore the data a bit.

# width and length of sepals/petals in cm
TITLES = ["sepal length", "sepal width", "petal length", "petal width", "class"]

import pandas as pd
import numpy as np
from sklearn import metrics

df = pd.read_csv('data/iris.data', header=None, names=TITLES)

# replace "NA" with "NaN" for easier dropping
df = df.replace('NA', np.nan, regex=True)
df = df.dropna()

df_binary = df.drop(columns=["class"])
print(df_binary)

### Ex2. Apply a K-means clustering to the iris data.
# Choose the n-clusters parameter suitably. Ignore the class column.
# You can simply fit_predict the KMeans to the data in this case, no need to split or anything.
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# The process of transforming numerical features to use the same scale is known as feature scaling.
# It’s an important data preprocessing step for most distance-based machine learning algorithms because
# it can have a significant impact on the performance of your algorithm.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_binary)
# print(scaled_features)

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)

print("Final locations of the centroid: ", kmeans.cluster_centers_)

predictions = kmeans.fit_predict(scaled_features)

print("FIT AND PREDICT: ", predictions)

### Ex3. Draw two scatterplots using two columns from the data as the x and y,
# you can choose the two columns freely except don't use the class column.
# In the first plot use the k-means clusters as the hue and in the second one use the class as the hue.

def ScatterPlots(local_predictions, titlename):
    # scatter plot with "sepal length", "petal length" and hue is based on k-means clusters
    plt.style.use("fivethirtyeight")
    plt.scatter(df["sepal length"], df["petal length"], s=200, c=local_predictions, cmap='Greens')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.title(titlename + ' Clusters (colored by predictions)')
    plt.show()

    # second scatterplot where class is the hue
    groups = df.groupby('class')
    for name, group in groups:
        plt.plot(group["sepal length"], group["petal length"], marker='o', linestyle='', markersize=12, label=name)

    plt.legend()
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.title(titlename + ' Clusters (colored by class)')
    plt.show()

ScatterPlots(predictions, "K-Means")

### Ex4. Test different clustering methods available in sklearn using the iris data.
# Does any of the methods give more similar clusters w.r.t. the iris classes compared to K-means?

from sklearn.cluster import AffinityPropagation
# creates clusters by sending messages between pairs of samples until convergence
sc = AffinityPropagation(random_state=5)
ScatterPlots(sc.fit_predict(scaled_features), "Affinity Propagation")

from sklearn.cluster import MeanShift
# aims to discover “blobs” in a smooth density of samples
sc = MeanShift(bandwidth=2)
ScatterPlots(sc.fit_predict(scaled_features), "Mean Shift")

from sklearn.cluster import DBSCAN
sc = DBSCAN(eps=3, min_samples=3)
ScatterPlots(sc.fit_predict(scaled_features), "DBSCAN")

answer = '''
I tested 3 other clustering methods. Affinity Propagation, Mean Shift, and DBSCAN. The closest with
respect to K-Means is Affinity Propagation I believe. However, none of these clustering methods were
as close to creating clusters that matched the iris classes like K-Means did.
'''
print(answer)

### Ex5. Explain where/when you would use a clustering algorithm.

answer = ''''
Clustering algorithms have multiple advantages and reasons to be used, and each one has different benefits
to being used. However, clustering algorithms are generally helpful when working with a large unstructured
data set, when looking for anomolies in the data. Where they are used is typically market research and
customer segmentation, when it's easy to make customers.
'''
print(answer)