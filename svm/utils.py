import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def get_iris():
    iris = datasets.load_iris()
    pos = np.array([[x[0], x[3]] for x in iris.data])
    label = np.array([1 if y == 0 else -1 for y in iris.target])
    # divide dataset into training set and test set
    train_index = np.random.choice(len(pos), int(len(pos) * 0.8), replace=False)
    test_index = np.array(list(set(range(len(pos))) - set(train_index)))
    # training dataset
    train_pos = pos[train_index]
    train_label = label[train_index]
    # test dataset
    test_pos = pos[test_index]
    test_label = label[test_index]
    return train_pos, train_label, test_pos, test_label


def create_dataset():
    pass