# -- coding: utf-8 --
"""
A module implements LinerRegression Algorithm.
$$$ Introduce:
        LinerRegression Algorithm is a liner approach to modeling the relationship between the dependent variable
        ans the independent variable.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class LinearRegression:
    """
    This is a class that implement the Ordinary Liner Regression Algorithm.

    Parameters
    --------------
        X: numpy.array
            The train data that is a two dimensions numpy array.
            for example:
             this is acceptable:
                X=numpy.array([[1],
                            [2]])
             and this is unacceptable
                X=numpy.array([1, 2])

        y: numpy.array
            It's the target data and also should be a two dimensions numpy array.

    Attributes
    --------------
        theta_:
            the parameters of the predicting liner regression equation.
            specially the theta_[0] is the constant of the equation.

    """
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.theta_ = None

    def GetLossValue(self, theta: np.array):
        return np.sum((self.__y - self.__X.dot(theta)) ** 2) / len(self.__X)

    def DLossFun(self, theta: np.array):
        return self.__X.T.dot(self.__X.dot(theta) - self.__y) * 2 / len(self.__X)

    def fitByGradintDescent(self, init: np.array, maxloop: int=10000000, sep: float=0.00001, eps: float=1e-10):
        import GradientDescent
        gradientdescent = GradientDescent.GradientDescent(self.GetLossValue, self.DLossFun)
        gradientdescent.search(init=init, maxloop=maxloop, sep=sep, eps=eps)
        self.theta_ = gradientdescent.ans_

    def StochasticDLossFun(self, theta: np.array, index: int):
        tmp = self.__X[index].reshape(-1, 1) * (self.__X[index].dot(theta) - self.__y[index])
        return tmp
# t0越高混乱程度越高，t1越低混乱程度越高
    def fitByStochasticGradintDescent(self, init_theta: np.array, t1: float=50.0, t0: float=5, n_iters:int=1000):
        import GradientDescent
        stdgradescent = GradientDescent.StochasticGradientDescent(getFunValue=self.GetLossValue, DFun=self.StochasticDLossFun)
        stdgradescent.search(length=len(self.__X), init_theta=init_theta, t1=t1, t0=t0, n_iters=n_iters)
        self.theta_ = stdgradescent.ans_

    def fitByNormalEquation(self):
        tmp = self.__X[:, 1:]
        self.theta_ = np.linalg.inv(tmp.T.dot(tmp)).dot(tmp.T).dot(self.__y)

    def predicts(self, precomputes: np.array):
        return np.array([self.__predictSingle(i) for i in precomputes])

    def __predictSingle(self, precompute: np.array):
        return precompute.dot(self.theta_)

    def score(self, X_test, y_test):
        return 1 - np.sum((np.reshape(self.predicts(X_test), (-1, 1)) - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

"""
from ModelSelection import trainTestSplit
from Scaler import StandardScaler

bost = datasets.load_boston()
data = bost.data[:, 5]
target = bost.target
data = data[target < np.max(target)].reshape(-1, 1)
target = target[target < np.max(target)].reshape(-1, 1)

X_train, X_test, y_train, y_test = trainTestSplit(X_data=data, y_target=target, seeds=6660)

standardscaler = StandardScaler()
standardscaler.fit(X_train)
X_train = standardscaler.transForm(X_train)
X_test = standardscaler.transForm(X_test)

liner = LinearRegression(X_train, y_train)
plt.scatter(X_train, y_train)
liner.fitByStochasticGradintDescent(init_theta=np.zeros((2, 1)))
# liner.fitByNormalEquation()
plt.plot([np.min(X_train), np.max(X_train)], [np.min(X_train) * liner.theta_[1] + liner.theta_[0], np.max(X_train) * liner.theta_[1] + liner.theta_[0]], color='r')
plt.show()
print(liner.score(X_test, y_test))

"""