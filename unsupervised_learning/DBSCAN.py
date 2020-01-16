from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

"""
This is a playground for the DBSCAN algorithm
Iwould try epsilon 0.13 and min_sample 5    
"""


"""ploting result """
def display_graph(X, y):
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

#generate dataset
X, y = make_moons(n_samples=500, noise=0.1)

#find a good epsilon using distance graph
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.title('distances')
plt.plot(distances)
plt.show()


eps = input("epsilon  :\n")
min_samples = input("min_samples  :\n")

#define epsilon neighbourhood and minimum sample in the neighbourhood based on the result
dbscan = DBSCAN(eps=float(eps), min_samples=float(min_samples))
dbscan.fit(X)
#Here are the labels
clusters = dbscan.labels_
#np.unique(clusters)
#dbscan.core_sample_indices_.shape
#dbscan.components_.shape


colors = ['olive', 'maroon','royalblue',  'forestgreen', 'mediumorchid', 'tan', 'deeppink',  'goldenrod', 'lightcyan', 'navy']
#colors = { 0:'green', 1:'yellow'}
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.title('input data')
plt.scatter(X[:,0], X[:,1])
plt.show()
plt.title('output data')
plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
plt.show()
