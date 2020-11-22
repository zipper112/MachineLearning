# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ModelSelection import trainTestSplit
from Scaler import StandardScaler


np.random.seed(666)
X = np.arange(-5, 15, 0.2).reshape(-1, 1)
y = X * X + 5 + (np.random.random(size=X.shape) - 0.5) * 50
poly = PolynomialFeatures(degree=15)
poly.fit(X)
X = poly.transform(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transForm(X)
scaler.fit(y)
y = scaler.transForm(y)


def showLearningCurve(data: np.array, target:np.array, train_ratio: float=0.7):
    X_train, X_test, y_train, y_test = trainTestSplit(data, target, train_ratio=train_ratio)

    train_score = []
    test_score = []
    for i in range(len(X_train)):
        liner = LinearRegression()
        liner.fit(X_train[:i + 1], y_train[: i + 1])
        train_score.append(liner.score(X_train[: i + 1], y_train[: i + 1]) * 50)
        test_score.append(liner.score(X_test, y_test) * 50)

    plt.plot([i + 1 for i in range(len(train_score))], train_score, label='train')
    plt.plot([i + 1 for i in range(len(train_score))], test_score, label='test')
    plt.legend()
    plt.ylim(40, 50)
    plt.show()
