from scipy import sparse as sp
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

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


# variance ration

colors = ['olive', 'maroon','royalblue',  'forestgreen', 'mediumorchid', 'tan', 'deeppink',  'goldenrod', 'lightcyan', 'navy']
#colors = { 0:'green', 1:'yellow'}
#vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
#plt.scatter(X_test[:, 0], y_test[:, 0], c=vectorizer(y_train_partially_propagated), s=50, cmap='viridis')
plt.plot(pca.explained_variance_ratio_)
