import utils

import numpy as np
import tensorflow as tf

filename = 'train'

# create and get data
utils.create_dataset(filename)
data, n_samples = utils.read_dataset(filename)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()

# create weights and biases
w = tf.get_variable('weight', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# create network
y_pred = x * w + b

# define loss function -- squared error
loss = tf.square(y - y_pred, name='loss')

# gradient descent with learning rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# deploy
with tf.Session() as sess:
    # initialize w, b
    sess.run(tf.global_variables_initializer())
    # epoch = 100
    for i in xrange(100):
        sess.run(iterator.initializer)  # initialize iterator
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
        
        print 'Epoch %d: loss: %f' % (i, total_loss / n_samples)
    # get result
    w_out, b_out = sess.run([w, b])
    print 'training result: y = %sx + %s' %(w_out, b_out)
    # visualize result
    pred_y = [w_out * datax + b_out for datax in data[:,0]]
    utils.plot(data[:, 0], data[:, 1], pred_y)

