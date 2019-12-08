import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()  # for plot styling
X = pd.read_csv("data/creditcard.csv")

from sklearn.cluster import KMeans
#define the number of cluster
k = 4
kmeans = KMeans(n_clusters=k)

print(X)

#reshape in 2D array
#nsamples, nx, ny = X.shape
#X = X.reshape((nsamples,nx*ny))


y_pred = kmeans.fit_predict(X)
kmeans.cluster_centers_

#assign new data to the closest cluster centroid
#X_new = np.array([0,2])
#kmens.predict(X_new)
y_kmeans =y_pred

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#source https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
