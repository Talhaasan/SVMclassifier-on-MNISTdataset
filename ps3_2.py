from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn import metrics

x, y = fetch_openml('mnist_784',return_X_y=True)

x = x / 255.0

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]


model_linear = svm.SVC(kernel='rbf',degree=3)
model_linear.fit(x_train, y_train)

y_pred = model_linear.predict(x_test)

print("Classification Score:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))