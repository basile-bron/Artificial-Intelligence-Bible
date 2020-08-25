import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/weatherHistory.csv")

X= data["Temperature (C)"]
y = data["Humidity"]
xy = X*y
x2 = X**2

b = (xy.sum()-X.sum()*y.sum()/len(X)) / (x2.sum()-X.sum()**2/len(X))
a = y.sum()/len(X)-b*X.sum()/len(X)
y_p = a + b*X
print('Fitted regression: y = '+str(a)+' + '+str(b)+'x')

plt.scatter(X, y)
plt.plot(X, y_p, color='orange')

data.plot(kind='hexbin',
            x='Temperature (C)',
            y='Humidity',
            gridsize=20, figsize=(12,8),
            cmap="Blues", sharex=False)

plt.plot(X, y_p, color='orange')
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
