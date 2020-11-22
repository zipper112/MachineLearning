
# -- coding: utf-8 --
import numpy as np
from collections import Counter
from sklearn import datasets
import matplotlib.pyplot as plt

class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.X_data = None
        self.y_target = None

    def fit(self, X_tarin: np.array, y_train: np.array):
        self.X_data = X_tarin
        self.y_train = y_train

    def __single_predict(self, x: np.array):
        distances = np.sum((self.X_data - x) ** 2, axis=1)
        k_neighbors = np.argsort(distances)[: self.k]
        counter = Counter(self.y_train[k_neighbors])
        return counter.most_common()[0][0]

    def predicts(self, X_predicts):
        res = [self.__single_predict(i) for i in X_predicts]
        return np.array(res)

    def score(self, X_predicts: np.array, y_target: np.array):
        return np.sum(self.predicts(X_predicts) == y_target) / len(y_target)

# 这里y用行向量
from Scaler import StandardScaler