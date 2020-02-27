
from sklearn.dataset import load_digits

#split training and test set
from sklearn.model_selection import tran_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

#fiting a logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = logisticRegression()
log_reg.fit(X_train, y_train)

#evaluate accuracy
log_reg.score(X_test, y_test)
###########################

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression()),
])

pipeline.fix(X_test, y_train)

pipeline.score(X_test, y_test)
