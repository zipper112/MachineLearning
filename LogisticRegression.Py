import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from GradientDescent import GradientDescent
from GradientDescent import StochasticGradientDescent
from ModelSelection import trainTestSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class LogisticRegression:

    def __init__(self):
        self.intercept_ = None
        self.theta_ = None
        self.data = None
        self.target = None

    def fit(self, data: np.array, target: np.array, method='GD'):
        from Scaler import decorate
        self.data = decorate(data)
        self.target = target.reshape(-1, 1)
        if method == 'GD':
            gd = GradientDescent(getFunValue=self.__LossFun, DFun=self.DFun)
            init = np.ones(shape=(self.data.shape[1], 1))
            gd.search(init=init, maxloop=60000, sep=0.05, eps=1e-6)
            self.intercept_ = gd.ans_[0]
            self.theta_ = gd.ans_
        elif method == 'SD':
            sd = StochasticGradientDescent(getFunValue=self.__LossFun, DFun=self.DFun)
            sd.search(init_theta=np.ones(shape=(self.data[1], 1)), length=self.data.shape[0])
            self.intercept_ = sd.ans_[0]
            self.theta_ = sd.ans_
        else:
            raise Exception('KeyWord Error:[{}]'.format(method) + "can't be use to fit this object.")
        return self

    def __LossFun(self, theta):
        y_hat = self.__Sigmoid(self.data.dot(theta))
        return -(self.target.T.dot(y_hat) + np.log1p(1 - self.__Sigmoid(y_hat))) / self.data[0]

    def __Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def DFun(self, theta):
        return self.data.T.dot(self.__Sigmoid(self.data.dot(theta)) - self.target) / self.data.shape[0]

    def predict(self, data_predict):
        from Scaler import decorate
        pre = decorate(data_predict)
        res = np.array(self.__Sigmoid(pre.dot(self.theta_)) >= 0.5, dtype='int')
        return res.T[0]

    def score(self, data_predict, target_predict):
        return np.sum(self.predict(data_predict).T == target_predict) / data_predict.shape[0]


def ShowBound(model, data, target, categories):
    from matplotlib.colors import ListedColormap
    axis = [np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])]

    lst = ListedColormap(['#98F5FF', '#54FF9F', '#EEE685'])
    x, y = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) / 0.01)),
                       np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) / 0.01)))
    pre = np.c_[x.ravel(), y.ravel()]

    pre_target = model.predict(pre)
    plt.contourf(x, y, pre_target.reshape(x.shape), alpha=0.5, cmap=lst)
    for i in range(categories):
        plt.scatter(data[target == i][:, 0], data[target == i][:, 1])
    plt.xlim(axis[0], axis[1])
    plt.ylim(axis[2], axis[3])

# 制造数据集------------


def main():
    # np.random.seed(123)
    # data = np.random.random(size=(400, 2)) * 3
    # target = np.array(5 * (data[:, 0] - 1.5) ** 2 + np.random.random(size=data[:, 0].shape)
    #                   <= (data[:, 1] - 1.5) ** 2, dtype='int')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    iris = datasets.load_digits()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = trainTestSplit(data, target, train_ratio=0.7)
    log_reg = Pipeline(
        [('pca', PCA(tol=0.95)),
         ('poly', PolynomialFeatures(degree=1)),
         ('scaler', StandardScaler()),
         ('logistic_regression', LogisticRegression(penalty='l2', multi_class='ovr', C=1))]
    )
    from PolynomialRegression import showLearningCurve
    showLearningCurve(log_reg, sep=50, data=data, target=target)
    print(log_reg.score(X_test, y_test))


if __name__ == '__main__':
    main()

"""
from sklearn.preprocessing import StandardScaler
logic_reg = Pipeline(
    [('poly', PolynomialFeatures(degree=2)),
     ('scaler', StandardScaler()),
     ('logistic_regression', LogisticRegression())]
)
logic_reg.fit(X_train, y_train)

# 绘制曲线&评估模型----------------------------


ShowBound(logic_reg, [np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])])
# pre = logic_reg.predict(X_test[:, 1:]).T[0]
plt.scatter(data[target == 0][:, 0], data[target == 0][:, 1])
plt.scatter(data[target == 1][:, 0], data[target == 1][:, 1])
plt.xlim(np.min(data[:, 0]), np.max(data[:, 0]))
plt.ylim(np.min(data[:, 1]), np.max(data[:, 1]))
plt.show()
print(logic_reg.score(X_test, y_test))

"""

