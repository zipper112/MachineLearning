import pandas as pd 
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# x = pd.read_csv('train.csv')
# data = x[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
# target = x['Survived']

# def ExtendFeture(key: str, x: pd.DataFrame):
#     for i in set(x[key]):
#         if pd.isna(i):
#             continue
#         data[str(i)] = pd.Series(x.loc[:, key] == i, dtype=int)
#     x = x.drop(key, axis=1)
#     return x

# data = ExtendFeture('Sex', data)
# data = ExtendFeture('Pclass', data)
# data = ExtendFeture('Embarked', data)
# print(data)
digits = sklearn.datasets.load_digits()
data = digits['data']
target = digits['target']
target[target >= 2] = 0

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.7, random_state=666)
log_reg = Pipeline([
    ('std', StandardScaler()),
    ('log_reg', LogisticRegression(penalty='l2', C=5))
])
log_reg.fit(X_train, y_train)
# def getTN(model, data, target):
#     return np.sum(model.predict(data[target == 1]) == 1)

# def getPrecision(model, data, target):
#     s = np.sum(model.predict(data) == 1)
#     if not s:
#         return 0
#     return getTN(model, data, target) / s

# def getRecall(model, data, target):
#     s = np.sum(target == 1)
#     return getTN(model, data, target) / s
print(precision_score(y_test, log_reg.predict(X_test)))