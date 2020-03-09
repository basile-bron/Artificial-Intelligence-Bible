from scipy import sparse as sp
import numpy as np
from sklearn.datasets import load_digits
X, Y = load_digits(return_X_y=True)

#split training and test set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)


X_centered = X - X.mean(axis=0)

# WITH NUMPY
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

# WITH SCIKIT-LEARN
from sklearn.decomposition import PCA

pca = PCA(n_components =2)
X2D = pca.fit_transform(X)
