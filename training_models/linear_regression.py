import numpy as np
import matplotlib.pyplot as plt

#generate random dataset
X = 2 * np.random.rand(100,1)
#function : y = 4 + 3 *X + random gaussian noise
y = 4 + 3 * X + np.random.randn(100,1)
plt.plot(X,y, 'ro')
plt.show()

#linear regression
X_b = np.c_[np.ones((100, 1)), X] #add x0 = 1 to each instances because we don't want to multiply by less than 1
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("theta best :", theta_best)

# make predictions using theta_best
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print("y_predict best :", y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

###################################################
#USING SCIKIT LEARN
###################################################
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
y_predict = lin_reg.predict(X_new)


theta_best_svd, residuals, rank , s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print("theta best :", theta_best_svd)

f = np.linalg.pinv(X_b).dot(y)
print("y_predict best :", y_predict)

#PLOT
plt.plot(X_new, y_predict, "r-")
plt.plot(X_new, f, "b-")
plt.plot(X, y, "b.")
#plt.axis([0, 2, 0, 15])
plt.show()
