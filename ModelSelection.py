import numpy as np


def trainTestSplit(X_data: np.array, y_target: np.array, seeds: int=114514, shuffle: bool=True, train_ratio: float=0.8):
    shuffle_indexes = np.arange(len(y_target))
    train_size = int(len(y_target) * train_ratio)
    if shuffle:
        np.random.seed(seeds)
        np.random.shuffle(shuffle_indexes)
    return X_data[shuffle_indexes[:train_size]], X_data[shuffle_indexes[train_size:]], y_target[shuffle_indexes[:train_size]], y_target[shuffle_indexes[train_size:]]
