import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def get_iris():
    """
    iris dataset for svm implemented without kernel (linear seperate)
    """
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
    """
    dataset for svm implemented by svm
    """
    data, label = datasets.make_circles(n_samples=350, noise=0.1, factor=0.5)
    label = np.array([1 if l == 1 else -1 for l in label]).reshape(len(data), 1)
    return data, label


def show_dataset(data, label):
    set1x = [d[0] for i, d in enumerate(data) if label[i] == -1]
    set1y = [d[1] for i, d in enumerate(data) if label[i] == -1]
    set2x = [d[0] for i, d in enumerate(data) if label[i] == 1]
    set2y = [d[1] for i, d in enumerate(data) if label[i] == 1]
    plt.plot(set1x, set1y, 'o', color='red')
    plt.plot(set2x, set2y, 'o', color='blue')