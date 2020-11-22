# -- coding: utf-8 --
"""
A module used to help learning PCA.
$$$ Introduce:
        PCA
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def demean(data: np.array):
    return data - np.mean(data, axis=0)


def standard(w: np.array):
    return w / np.sum(w ** 2) ** 0.5


class PCA:
    def __init__(self, n_component=2, ratios=0.95):
        self.n_component = n_component
        self.components_ = None
        self.ratios = ratios
        self.vars = None

    def funValue(self, data: np.array, w: np.array):
        return np.sum(data.dot(w) ** 2) / len(data)

    def dFun(self, data: np.array, w: np.array):
        return data.T.dot(data.dot(w)) * 2 / len(data)

    def gradientAscent(self, X: np.array, eps: float = 1e-8, maxloop: int = 60000,
                       sep: float = 100):
        self.vars = np.empty(shape=X.shape[1])  # 方差数组
        X_pca = demean(X)  # demean操作
        w_d = np.empty(shape=(X.shape[1], X.shape[1]))  # 存主成分

        for i in range(X.shape[1]):
            w = standard(np.ones(shape=(X.shape[1], 1)))
            count = 0
            while True:
                next = standard(w + sep * self.dFun(w=w, data=X_pca))
                if abs(self.funValue(X_pca, w) - self.funValue(X_pca, next)) < eps:
                    w_d[i] = next.T
                    self.vars[i] = self.funValue(X_pca, next)
                    break
                w = next
                count += 1
                if count == maxloop:
                    return None
            X_pca = demean(X_pca - X_pca.dot(w_d[i].reshape(-1, 1)) * w_d[i])
        self.components_ = w_d

    def __transform(self, X: np.array, k: int):
        return X.dot(self.components_[: k].T)

    def reTransForm(self, X: np.array):
        return X.dot(self.components_[: self.n_component])

    def transFormByK(self, X: np.array):
        return self.__transform(X, self.n_component)

    def transFormByratios(self, X: np.array):
        percents = self.vars / np.sum(self.vars)
        k, sum = -1, 0
        while sum <= self.ratios and k < len(self.vars):
            sum += percents[k + 1]
            k += 1
        return self.__transform(X, k)

