# -- coding: utf-8 --
import numpy as np


def decorate(X: np.array):
    return np.hstack([np.ones((len(X), 1)), X])


class StandardScaler:
    def __init__(self):
        self.E = None
        self.std = None

    def fit(self, X: np.array):
        self.E = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[np.abs(self.std) < 1e-14] = 1

    def transForm(self, X: np.array):
        return (X - self.E) / self.std

class NormalScaler:
    def __init__(self):
        self.maximum = None
        self.minimum = None

    def fit(self, X: np.array):
        self.maximum = np.max(X)
        self.minimum = np.min(X)

    def transForm(self, X: np.array):
        return (X - self.minimum) / (self.maximum - self.minimum)

