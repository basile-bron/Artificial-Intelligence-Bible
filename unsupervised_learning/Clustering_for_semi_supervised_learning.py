from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as np
X_digits, y_digits = load_digits(return_X_y=True)

#split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

# CLUSTERING FOR SEMI SUPERVISED LEARNING
n_labeled = 50
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
#performance of the model on the test set ?
log_reg.score(X_test, y_test)

#### Tag some of the imaes
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

#display images
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

X_representative_digits.shape
img = X_representative_digits.reshape([50  ,8, 8])
show_images(img[0:50], cols = 1, titles = None)

y_representative_digits = np.array([4,0,7,6,8,9,5,2,3,3,8,4,9,2,5,9,7,0,6,1,2,8,4,5,3,4,6,4,8,5,6,1,2,7,9,1,1,2,1,4,0,9,2,2,7,7,9,4,4,8])
#################
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=1000)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)


#LABEL PROPAGATION
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

#training the model and see performance result
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
