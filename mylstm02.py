import tensorflow as tf
import numpy as np

import gensim





lr = 0.001
training_iters = 100000
batch_size = 128
num_labels = 3
n_inputs = 28   #  data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      #  classes (0-9 digits)




label = tf.placeholder(tf.float32, [None, 3])
input_data = tf.placeholder(tf.int32, [None, max_tokens])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([num_labels, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, num_labels]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[num_labels, ]))
}

def lstm(x, weight, bias) :
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pred = lstm(input_data, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=train_label))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(test_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12

    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            input_data: train_text,
            labels: test_label,
        })

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            input_data: train_text,
            labels: test_label,
            }))
        step += 1
