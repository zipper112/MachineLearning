from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import PolynomialFeatures
from LogisticRegression import ShowBound
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# minst = fetch_mldata('MNIST original', data_home=r'D:\datasets')

np.random.seed(123)
data = np.random.random(size=(400, 2)) * 3
target = np.array(5 * (data[:, 0] - 1.5) ** 2 + np.random.random(size=data[:, 0].shape) * 2
                     <= 3 * (data[:, 1] - 1.5) ** 2, dtype='int')

log_reg = Pipeline([
    ('ploy', PolynomialFeatures(degree=30)),
    ('std', StandardScaler()),
    ('log', LogisticRegression(C=2))
])
log_reg.fit(data, target)
ShowBound(log_reg, data, target, 2)
plt.show()
