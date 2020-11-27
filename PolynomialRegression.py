# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ModelSelection import trainTestSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

# -----数据生成-----
np.random.seed(666)
X = np.arange(-5, 15, 0.1).reshape(-1, 1)
y = X * X + 5 + (np.random.random(size=X.shape) - 0.5) * np.random.randint(low=50, high=100, size=X.shape)

# -----学习曲线绘制-----


def showLearningCurve(data: np.array, target:np.array, train_ratio: float=0.7):
    poly = Pipeline(
        [("poly", PolynomialFeatures(degree=2)),
         ("scaler", StandardScaler()),
         ("liner", LinearRegression())]
    )
    X_train, X_test, y_train, y_test = trainTestSplit(data, target, train_ratio=train_ratio)

    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        poly.fit(X_train[:i], y_train[: i])
        train_score.append(poly.score(X_train[: i], y_train[: i]))
        test_score.append(poly.score(X_test, y_test))

    plt.plot([i + 1 for i in range(len(train_score))], train_score, label='train')
    plt.plot([i + 1 for i in range(len(train_score))], test_score, label='test')
    plt.legend()
    plt.ylim(-3, 1)
    plt.show()