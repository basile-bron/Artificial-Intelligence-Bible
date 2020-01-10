import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans

def k_mean(nbr_cluster,data):
    #define the number of cluster
    k = nbr_cluster
    kmeans = KMeans(n_clusters=k)
    y_kmeans = kmeans.fit_predict(data)
    return y_kmeans
    #print(X)
    #kmeans.cluster_centers_

    #reshape in 2D array
    #nsamples, nx, ny = X.shape
    #X = X.reshape((nsamples,nx*ny))

#assign new data to the closest cluster centroid
#X_new = np.array([0,2])
#kmens.predict(X_new)

def plot(y_kmeans, data):
    data = pd.DataFrame.to_numpy(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#source https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
def main(nbr_cluster, path):
    data = pd.read_csv(path)
    y_kmeans = k_mean(nbr_cluster,data)
    plot(y_kmeans,data)


main(31, "data/winequality-red.csv")
main(7, "data/winequality-white.csv")
