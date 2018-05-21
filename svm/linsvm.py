import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

# parameter
batch_size = 100
beta = 0.01
epoch = 5000
learning_rate = 0.01

# get data
train_pos, train_label, test_pos ,test_label = utils.get_iris()

# create placeholder
X = tf.placeholder(tf.float32, name='X')    
Y = tf.placeholder(tf.float32, name='Y')
L = tf.placeholder(tf.float32, name='L')

# create weights and bias
K = tf.get_variable('k', initializer=tf.random_normal([2, 1]))          # (2,1)
b = tf.get_variable('b', initializer=tf.constant(0.0, shape=[1, 1]))    # (1,1)

# create model
in1 = tf.concat([X, Y], 1)                  # (batch_size, 2)
output = tf.matmul(in1, K) - b              # (batch_size, 1)

# L2 norm
l2_norm = tf.reduce_sum(tf.square(K))
loss = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(output, L))))\
        + beta * l2_norm

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
prediction = tf.sign(output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, L), tf.float32))

# deploy
with tf.Session() as sess:
    # initialize global weights
    sess.run(tf.global_variables_initializer())
    for epo in xrange(epoch):
        sample_index = np.random.choice(len(train_pos), batch_size)
        sample_x = np.asarray(train_pos[sample_index][:,0]).reshape((batch_size, 1))
        sample_y = np.asarray(train_pos[sample_index][:,1]).reshape((batch_size, 1))
        sample_l = np.asarray(train_label[sample_index]).reshape((batch_size, 1))
        _, acc = sess.run([optimizer, accuracy], feed_dict={X:sample_x, Y:sample_y, L:sample_l})

        if epo % 1000 == 0:
            print 'Epoch[%d]: training accuracy = %.1f%%' % (epo, acc * 100) 
    
    print 'testing...'
    test_x = np.asarray(test_pos[:, 0]).reshape((len(test_pos), 1))
    test_y = np.asarray(test_pos[:, 1]).reshape((len(test_pos), 1))
    test_label = test_label.reshape((len(test_label), 1))
    acc = sess.run(accuracy, feed_dict={X: test_x, Y: test_y, L: test_label})
    print 'Test result: accuracy: %.1f%%' % (acc * 100)

    # show training result
    [[a1], [a2]] = sess.run(K)
    slope = - a1 / a2
    [b] = sess.run(b)
    b = b / a2
    x = np.linspace(2, 10, 100)
    y = slope * x + b
    plt.plot(x, y, color='red')
    # show trainning dataset
    set1x = [d[0] for i, d in enumerate(train_pos) if train_label[i] == 1]
    set1y = [d[1] for i, d in enumerate(train_pos) if train_label[i] == 1]
    plt.plot(set1x, set1y, 'o', color='blue')
    set2x = [d[0] for i, d in enumerate(train_pos) if train_label[i] == -1]
    set2y = [d[1] for i, d in enumerate(train_pos) if train_label[i] == -1]
    plt.plot(set2x, set2y, 'o', color='yellow')
    plt.show()