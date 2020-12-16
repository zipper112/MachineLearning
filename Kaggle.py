import pandas as pd 
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import logging

# --------------logging-------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
#---------------logging-------------
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
"""
def FeatureProcessing():
    s = pd.read_csv('train.csv')
    target = s['Survived']
    data = s[['Sex', 'Pclass', 'Age', 'Fare', 'Embarked']]

    def discretization(x, label, ignore_nan=True):
        for i in set(x[label].values):
            if pd.isna(i) and ignore_nan:
                continue
            x[label + str(i)] = pd.Series(x[label] == i, dtype=int)
        x.drop(label, axis=1, inplace=True)
    discretization(data, 'Sex')
    discretization(data, 'Pclass')
    discretization(data, 'Embarked')
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    return data, target


def main():
    try: 
        data, target = FeatureProcessing()
        std = StandardScaler()
        std.fit(data)
        data = std.transform(data)
        svc = SVC()
        kf = KFold(n_splits=4, random_state=666)
        parameters = [
            {'C': np.linspace(0.1, 2, 10),
            'kernel': ['poly'],
            'gamma': np.linspace(0.1, 2, 5),
            'coef0': [0, 1, 2],
            'degree': [2, 3, 4]},
            {'C': np.linspace(0.1, 2, 10),
            'kernel': ['rbf'],
            'gamma': np.linspace(0.01, 0.1, 15),
            'decision_function_shape': ['ovo']}
            ]
        model = GridSearchCV(svc, parameters, cv=kf, n_jobs=-1)
        model.fit(data, target)
        svc = model.best_estimator_
        with open('model.tex', 'w') as stream:
            stream.write(str(model.best_score_) + '\n')
            stream.write(str(model.best_estimator_.__str__()))
    except Exception as a:
        logging.error(a.__str__)

if __name__ == '__main__':
    main()