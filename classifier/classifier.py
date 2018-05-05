import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

# parameter
K = 3                       # class
Epoch = 10000               # epoch
Learning_rate = 0.001       # learning rate
beta = 0.001                # L2 regularization

# create and get data
utils.create_dataset()
dataset, label, _ = utils.read_dataset('train') # dataset and label are np arrays
labels = (np.arange(K) == label[:,None]).astype(np.float32)

# create placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
L = tf.placeholder(tf.float32, name='L')

# create weights and biases
hidden_layer1_num = 64
hidden_layer2_num = 32
w1 = tf.get_variable('w1', initializer=tf.random_normal([2, hidden_layer1_num], stddev=2))
b1 = tf.get_variable('b1', initializer=tf.constant(0.0, shape=[1, hidden_layer1_num]))
w2 = tf.get_variable('w2', initializer=tf.random_normal([hidden_layer1_num, hidden_layer2_num], stddev=2))
b2 = tf.get_variable('b2', initializer=tf.constant(0.0, shape=[1, hidden_layer2_num]))
w3 = tf.get_variable('w3', initializer=tf.random_normal([hidden_layer2_num, K], stddev=2))
b3 = tf.get_variable('b3', initializer=tf.constant(0.0, shape=[1, K]))

# create network
in1 = tf.transpose(tf.concat([X, Y], 0))    # (n,2)
theta1 = tf.matmul(in1, w1) + b1            # (n,hidden_1)
h1 = tf.nn.relu(theta1)                     # (n,hidden_1)
theta2 = tf.matmul(h1, w2) + b2             # (n,hidden_2)
h2 = tf.nn.relu(theta2)                     # (n, hidden_2)
theta3 = tf.matmul(h2, w3) + b3             # (n, K)
pred = tf.nn.softmax(theta3, axis=1)        # (n, K)

# loss: cross entropy
regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=L, logits=theta3))\
       + beta * regularization

# gradient descent with learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=Learning_rate).minimize(loss)

# define accuracy function
def accuarcy(_pred, _labels):
    return 100 * np.sum(np.argmax(_pred, 1) == np.argmax(_labels, 1)) / _pred.shape[0]

# deploy
with tf.Session() as sess:
    # initialize w1 - w3, b1 - b3
    sess.run(tf.global_variables_initializer())
    for epoch in xrange(Epoch):
        _, prediction = sess.run([optimizer, pred],\
                        feed_dict={X:[dataset[:,0]], Y:[dataset[:,1]], L:labels})
        if epoch % 1000 == 0:
            print 'Epoch[%d]: accuracy = %.1f%%' % (epoch, accuarcy(prediction, labels))

    # show classifier -- view training result
    utils.plot(dataset, np.argmax(prediction, axis=1), True)