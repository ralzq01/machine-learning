# implemtent svm with kernels

# SVM Lagrangian problem
#       minimize_{w, b} \frac{1}{2}||w||^2
#       subject to y_i(w * x_i + b) - 1 >= 0, i = 1,...,m

# SVM Wolfe dual problem
#       maximize_{\alpha}\ \Sigma_{i=1}^{m}\alpha_i - \frac{1}{2}\Sigma_{i=1}^{m}\Sigma_{j=1}^{m}\alpha_i\alpha_jy_iy_jx_ix_j
#       subject to \alpha_i >= 0 for any i = 0,...,m
#                  \Sigma_{i=1}^{m}\alpha_iy_i
#       after getting mutiplier \alpha, we can calculate w, b:
#       w = \Sigma_{i=1}^{m} \alpha_iy_ix_i
#       b = \frac{1}{S}\Sigma_{i=1}^{S} (y_i - wx_i)

# SVM Kernel options
#
# Gaussian Kernel:
#   K(x1, x2) = exp{-gamma||x1 - x2||^2}
#
# Polynomial Kernel:
#   K(x1, x2) = (x1x2 + c)^d

import numpy as np
import tensorflow as tf
import utils
import matplotlib.pyplot as plt

# parameter
learning_rate = 0.002
batch_size = 200
gamma = 50.
epoch = 1000

# get dataset
data, label = utils.create_dataset()

# create placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='X')    # (n,2)
L = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='L')    # (n,1)
P = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='P')    # (n,2)

# create model
alpha = tf.get_variable('a', initializer=tf.random_normal([batch_size, 1]))

# create kernel -- Gaussian kernel
dist = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])         # (n,1)
exponent = dist - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(dist)
kernel = tf.exp(-gamma * exponent)

# loss
first = tf.reduce_sum(alpha)
alpha_cross = tf.abs(tf.matmul(alpha, tf.transpose(alpha)))
label_cross = tf.matmul(L, tf.transpose(L))
second = 0.5 * tf.reduce_sum(alpha_cross * label_cross * kernel)
loss = second - first

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction
rA = dist
rB = tf.reshape(tf.reduce_sum(tf.square(P), 1), [-1, 1])
exponent_pred = rA - 2 * tf.matmul(X, tf.transpose(P)) + tf.transpose(rB)
kernel_pred = tf.exp(-gamma * exponent_pred)

prediction_output = tf.matmul(tf.transpose(kernel_pred), alpha * L)
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, L), tf.float32))

# deploy
with tf.Session() as sess:
    # initialize global weights
    sess.run(tf.global_variables_initializer())
    for epo in xrange(epoch):
        sample_index = np.random.choice(len(data), batch_size)
        sample_data = data[sample_index].reshape((batch_size, 2))
        sample_label = label[sample_index].reshape((batch_size, 1))
        sess.run(optimizer, feed_dict={X:sample_data, L:sample_label, P:sample_data})
        acc = sess.run(accuracy, feed_dict={X:sample_data, L:sample_label, P:sample_data})

        if (epo + 1) % 100 == 0:
            print 'Epoch[%d]: training accuracy = %.1f%%' % (epo + 1, acc * 100)
    
    # draw result
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 0].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = sess.run(prediction, 
                                feed_dict={X: sample_data,L: sample_label,P: grid_points})
    grid_predictions = grid_predictions.reshape(xx.shape)
    plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
    # draw dataset
    utils.show_dataset(data, label)

    plt.show()

