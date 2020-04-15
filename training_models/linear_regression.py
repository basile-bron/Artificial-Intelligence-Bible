import numpy as np
import matplotlib.pyplot as pt

#generate random dataset
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)
pt.plot(X,y, 'ro')

#
X_b = np.c_[np.ones((100, 1)), X] #add x0 = 1 to each instances
theta_best = np.linalg.inv(X_b.T.dot(X_b))

theta_best
