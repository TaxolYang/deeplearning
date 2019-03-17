import tensorflow as tf
import numpy as np
import gensim
from tensorflow.contrib.tensorboard.plugins import projector


from random import randint

import datetime


train_vec = np.load('train_index.npy')
train_label = np.load('train_label.npy')
test_vec = np.load('test_index.npy')

test_label = np.load('test_label.npy')

# print(test_vec)


lr = 0.001
training_iters = 100000
batch_size = 128

maxSeqLength = 29   # MNIST data input (img shape: 28*28)
word_dimension = 300    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 3      #  classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, maxSeqLength])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([maxSeqLength, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    for i in range(batch_size):
        # if (i % 2 == 0):
        #     num = randint(1,11499)
        #     labels.append([1,0])
        # else:
        #     num = randint(13499,24999)
        #     labels.append([0,1])
        arr[i] = test_vec[i]
        labels.append(train_label[i])
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    for i in range(batch_size):
        # num = randint(11499,13499)
        # if (num <= 12499):
        #     labels.append([1,0])
        # else:
        #     labels.append([0,1])
        arr[i] = test_vec[i]
        labels.append(train_label[i])
    return arr, labels

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


# tf.reset_default_graph()

# maxSeqLength = 29 #Maximum length of sentence
# numDimensions = 300 #Dimensions for each word vector
# # wordVectors = np.load(('wordVectors.npy'))
# batchSize = 24
# lstmUnits = 64
# numClass = 3
# iterations = 100000
# word_model = gensim.models.KeyedVectors.load_word2vec_format('D:/data/vec/google/GoogleNews-vectors-negative300.bin', binary=True)

# word_model = gensim.models.word2vec.Word2Vec.load('D:/data/vec/google/GoogleNews-vectors-negative300.bin')
# word_model = tf.convert_to_tensor(data)

word_model = np.load(('wordVectors.npy'))


# word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.txt')

print('加载word2vec')




def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.Variable(tf.zeros([batch_size, maxSeqLength, n_steps]),dtype=tf.float32)
    X = tf.nn.embedding_lookup(word_model,X)

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, maxSeqLength, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = getTrainBatch()
        batch_xs = batch_xs.reshape([batch_size, n_steps, maxSeqLength])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1
