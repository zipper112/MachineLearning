# -- coding: utf-8 --
import numpy as np


class GradientDescent:
    def __init__(self, getFunValue, DFun):
        self.getFunValue = getFunValue
        self.DFun = DFun
        self.ans_ = None

    def search(self, init: np.array, eps: float=1e-6, sep: float=0.1, maxloop: int=1000):
        tmp_res = init
        count = 0
        while True:
            next = tmp_res - self.DFun(tmp_res) * sep
            # print(abs(self.getFunValue(next) - self.getFunValue(tmp_res)))
            if abs(self.getFunValue(next) - self.getFunValue(tmp_res)) < eps:
                self.ans_ = next
                break
            count += 1
            if count == maxloop:
                break
            tmp_res = next
        return


class StochasticGradientDescent:
    def __init__(self, getFunValue, DFun):
        self.getFunValue = getFunValue
        self.DFun = DFun
        self.ans_ = None

    def search(self, init_theta: np.array, length: int, t1: float=50.0, t0: float=5.0, n_iters:int=5):
        tmp = init_theta
        cnt = 0

        def temper(x: int):
            return (t0 + 0.001 * x) / (x + t1)

        for i in range(n_iters):
            indexes = np.arange(length)
            np.random.shuffle(indexes)
            for j in indexes:
                tmp = tmp - self.DFun(theta=tmp, index=j) * temper(cnt)
                cnt += 1
        self.ans_ = tmp
        return
