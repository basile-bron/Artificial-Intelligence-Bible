from matplotlib.image import imread
from sklearn.cluster import KMeans
import os
image = imread(os.path.join("../data/landscape.jpg"))
print(image.shape)
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img = segmented_img.reshape(image.shape)
