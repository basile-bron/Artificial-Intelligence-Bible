from matplotlib.image import imread
from sklearn.cluster import KMeans
import os
import cv2
import matplotlib.pyplot as plt
import numpy as numpy

image_rgb = imread(os.path.join("C:/Users/Skull/Documents/github/Artificial-Intelligence-Lexical/data/landscape.jpg"))
#convert to HSV (apparently it is better than RGB for detection )
image_hsv=cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
#converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.
X = image_hsv.reshape(-1, 3)

#Kmean
kmeans = KMeans(n_clusters=6).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img = segmented_img.reshape(image_rgb.shape)
segmented_img.shape
#Print the image
# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(image_rgb)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('HSV image')
plt.imshow(image_hsv)

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('kmean result')
plt.imshow(segmented_img.astype('uint8'))
