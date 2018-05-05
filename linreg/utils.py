from matplotlib import pyplot as plt
import numpy as np

# create linear dataset y = WX + b
def create_dataset(filename):
    w = np.random.randint(-10, 10)
    b = np.random.randint(-10, 10)
    print 'creating dataset... y = %sx + %s' %(str(w), str(b))
    with open(filename, 'w') as file:
        # create 100 data sample
        for _ in xrange(100):
            x = 20 * np.random.ranf()
            delta_w = 2 * np.random.random_sample() - 1
            delta_b = 2 * np.random.random_sample() - 1
            y = (w + delta_w) * x + b + delta_b
            file.write(str(float(x)) + '\t' + str(float(y)) + '\n')

# read data from train file
def read_dataset(filename):
    with open(filename, 'r') as file:
        text = file.readlines()
        data = [line[:-1].split('\t') for line in text]
        x = [float(line[0]) for line in data]
        y = [float(line[1]) for line in data]
        data = list(zip(x, y))
        n_samples = len(data)
        data = np.asarray(data, dtype=np.float32)
    return data, n_samples
   
def plot(x, y, y_pred):
    plt.plot(x, y, 'bo', label='real data')
    plt.plot(x, y_pred, 'r', label='predicted data with squared error')
    plt.legend()
    plt.show()

