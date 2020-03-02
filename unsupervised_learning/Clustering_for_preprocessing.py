from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)

#split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

#fiting a logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
log_reg.fit(X_train, y_train)

#evaluate accuracy
log_reg.score(X_test, y_test)
###############################################
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=90)),
    ("log_reg", LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)),
])

pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)
###################################################
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2,100))
grid_clf = GridSearchCV(pipeline, {
    'alpha': [0.001, 0.0001], 'average': [True, False],
    'shuffle': [True, False], 'max_iter': [5], 'tol': [None]
}, param_grid, cv=3, verbose=1)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_
grid_clf.score(X_test, y_test)
