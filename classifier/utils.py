import numpy as np
import matplotlib.pyplot as plt

def create_dataset():
    # 3 classes on 2-D, 200 samples per classification
    N = 100
    K = 3
    data = np.zeros((N * K, 2))
    label = np.zeros((N * K), dtype='uint8')
    for j in xrange(K):
        k_index = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N)
        theta = np.linspace(j * 4.0, (j + 1) * 4.0, N) + np.random.randn(N) * 0.2
        data[k_index] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        label[k_index] = j
    with open('train', 'w') as file:
        for i in xrange(N * K):
            file.write('%f\t%f\t%s\n' % (data[i,0], data[i,1], label[i]))

def read_dataset(filename):
    with open(filename, 'r') as file:
        text = file.readlines()
        datatext = [line.split('\t') for line in text]
        X = [float(x[0]) for x in datatext]
        Y = [float(y[1]) for y in datatext]
        label = [int(l[2]) for l in datatext]
        dataset = zip(X, Y)
        n_samples = len(dataset)
        dataset = np.asarray(dataset, dtype=np.float32)
        label = np.array(label, dtype='uint8')
    return dataset, label, n_samples

def plot(dataset, label, show=False):
    color = ['blue', 'green', 'yellow']
    xlabel = [[],[],[]]
    ylabel = [[],[],[]]
    for i in range(len(dataset)):
        assert label[i] <= 2 and label[i] >= 0
        xlabel[label[i]] += [dataset[i, 0]]
        ylabel[label[i]] += [dataset[i, 1]]
    for i in xrange(3):
        plt.plot(xlabel[i], ylabel[i], 'bo', color=color[i])
    if show:
        plt.show()


if __name__ == '__main__':
    create_dataset()
    dataset, label, _ = read_dataset('train')
    plot(dataset, label, True)


