from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import pandas as pd

"""
This is a playground for the DBSCAN algorithm
Try epsilon 0.13 and min_sample 5
"""

#generate dataset
X, y = make_moons(n_samples=500, noise=0.1)

#uncomment to try with wine dataset
#X = pd.read_csv("../data/winequality-red.csv")

def distance_graph():
    #find a good epsilon using distance graph
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.title('distances')
    plt.plot(distances)
    plt.show()

distance_graph()
def run_DBSCAN(eps, min_samples, X):
    #distance_graph()
    eps = input("epsilon  :\n")
    min_samples = input("min_samples  :\n")

    #define epsilon neighbourhood and minimum sample in the neighbourhood based on the result
    dbscan = DBSCAN(eps=float(eps), min_samples=float(min_samples))
    dbscan.fit(X)
    #Here are the labels
    clusters = dbscan.labels_


    colors = ['olive', 'maroon','royalblue',  'forestgreen', 'mediumorchid', 'tan', 'deeppink',  'goldenrod', 'lightcyan', 'navy']
    #colors = { 0:'green', 1:'yellow'}
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

    try:
        X = pd.DataFrame.to_numpy(X)
    except:
        print("")
    plt.scatter(X[:, 0], X[:, 1], c=vectorizer(clusters), s=50, cmap='viridis')
    plt.show()

run_DBSCAN(0.13, 5, X)
