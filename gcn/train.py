import numpy as np
import pickle as pkl
import scipy.sparse as sp
import tensorflow as tf
import datetime
from gcn import GCN
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FLAGS = tf.app.flags.FLAGS

# define model
training = False
gcn_dims = [100, 85, 70, 53]
num_classes = 53
gcn_drop_prob = 0.5
batch_size = 320
# placeholder
features = tf.sparse_placeholder(dtype=tf.float32, name='features')
supports = tf.sparse_placeholder(dtype=tf.float32, name='adj')
num_features_nonzero = tf.placeholder(tf.int32, name='nonzero')
whole_label = tf.placeholder(dtype=tf.int32, shape=[None], name='whole_label')
not_NA_idx = tf.placeholder(dtype=tf.int32, shape=[None], name='not_NA_idx')
label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
index = tf.placeholder(dtype=tf.int32, name='index')
# gcn model
gcn = GCN(True, gcn_drop_prob, num_classes, gcn_dims)
out = gcn.gcn(features, supports, num_features_nonzero)
logits = tf.nn.embedding_lookup(out, index)
# loss
label_onehot = tf.one_hot(indices=label, depth=num_classes, dtype=tf.int32)
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=logits)
# optimizer
optimizer = tf.train.AdamOptimizer(0.05)
train_op = optimizer.minimize(loss)
# pred
pred = tf.cast(tf.argmax(out, axis=1), dtype=tf.int32)

def accuracy(label, pred):
    total = tf.cast(tf.size(label), dtype=tf.float32)
    match = tf.cast(tf.equal(label, pred), tf.float32)
    correct = tf.reduce_sum(match)
    return correct / total
# accuracy
acc = accuracy(whole_label, pred)
not_NA_label = tf.nn.embedding_lookup(whole_label, not_NA_idx)
not_NA_pred = tf.nn.embedding_lookup(pred, not_NA_idx)
not_NA_acc = accuracy(not_NA_label, not_NA_pred)

# load data
if training:
    print 'loading features...'
    load_features = np.load('data/train_eg.feature.npy')
    print 'loading adj...'
    with open('data/train_eg.adj', 'rb') as f:
        load_adj = pkl.load(f)
    print 'processing data...'
    load_features, load_adj = gcn.preprocess(load_features, load_adj)
    load_triples = np.load('data/train_instance_triple.npy')
else:
    print 'loading features...'
    load_features = np.load('data/test_eg.feature.npy')
    print 'loading adj...'
    with open('data/test_eg.adj', 'rb') as f:
        load_adj = pkl.load(f)
    print 'processing data...'
    load_features, load_adj = gcn.preprocess(load_features, load_adj)
    load_triples = np.load('data/test_instance_triple.npy')
load_pairs = load_triples[:,0:2] #(N,2)
load_label = load_triples[:,2].astype(np.int32) #(N,)
nonzero_idx = np.nonzero(load_label)[0]

# train
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        train_order = list(range(len(load_pairs)))
        np.random.shuffle(train_order)
        # train one step
        for i in range((int(len(train_order) / float(batch_size)))):
            batch_index = train_order[i * batch_size:(i + 1) * batch_size]
            # feed dict
            feed_dict = {
                features: load_features,
                supports: load_adj,
                num_features_nonzero:load_features[1].shape,
                index: batch_index,
                label: load_label[batch_index],
                whole_label: load_label,
                not_NA_idx: nonzero_idx
            }
            _, train_loss, train_not_NA_acc, train_acc = sess.run([train_op, loss, not_NA_acc, acc], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('epoch {e} step {s}: time: {time} | loss: {loss}, not NA accuracy: {Nacc}, total accraucy: {acc}'
                  .format(e=str(epoch), s=str(i), time=time_str, loss=str(train_loss), Nacc=str(train_not_NA_acc), acc=str(train_acc)))