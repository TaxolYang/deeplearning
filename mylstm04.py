import tensorflow as tf
import numpy as np
import gensim
from tensorflow.contrib.tensorboard.plugins import projector


from random import randint

import datetime

print('加载词向量')
train_vec = np.load('train_index.npy')
train_label = np.load('train_label.npy')

print('加载完成')
# print(train_vec)
# print(train_label)
# test_vec = np.load('test_index.npy')

# test_label = np.load('test_label.npy')

# print(test_vec)


def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        # if (i % 2 == 0):
        #     num = randint(1,11499)
        #     labels.append([1,0])
        # else:
        #     num = randint(13499,24999)
        #     labels.append([0,1])

        # print('词向量',i,'是',train_vec[i])
        # print('词向量',i,'是',train_label[i])

        arr[i] = train_vec[i]
        labels.append(train_label[i])
        # print('laberls',labels)
        # print('vec',arr)
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        # num = randint(11499,13499)
        # if (num <= 12499):
        #     labels.append([1,0])
        # else:
        #     labels.append([0,1])
        arr[i] = test_vec[i]
        labels.append(test_label[i])
    return arr, labels

# def get_W(word_vecs, k=300):
#     """
#     Get word matrix. W[i] is the vector for word indexed by i
#     """
#     vocab_size = len(word_vecs)
#     word_idx_map = dict()
#     W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
#     W[0] = np.zeros(k, dtype='float32')
#     i = 1
#     for word in word_vecs:
#         W[i] = word_vecs[word]
#         word_idx_map[word] = i
#         i += 1
#     return W, word_idx_map
def embeding_layer(vocab_size, embeding_size, inputs):
    '''
    Function used for creating word embedings (word vectors)

    Input(s): vocab_size - number of words in the vocab
              embeding_size - length of a vector used to represent a single word from vocab
              inputs - inputs placeholder

    Output(s): embed_expended -  word embedings expended to be 4D tensor so we can perform Convolution operation on it
    '''
    word_embedings = tf.Variable(tf.random_uniform([vocab_size, embeding_size]))
    embed = tf.nn.embedding_lookup(word_embedings, inputs)
    # embed_expended = tf.expand_dims(embed) #expend dims to 4d for conv layer
    return embed

tf.reset_default_graph()
vocab_size = 400000
lr = 0.001
maxSeqLength = 29 #Maximum length of sentence
numDimensions = 50 #Dimensions for each word vector
# wordVectors = np.load(('wordVectors.npy'))
batchSize = 24
lstmUnits = 64
numClass = 3
iterations = 100000
# word_model = gensim.models.KeyedVectors.load_word2vec_format('D:/data/vec/google/GoogleNews-vectors-negative300.bin', binary=True)

# word_model = gensim.models.word2vec.Word2Vec.load('D:/data/vec/google/GoogleNews-vectors-negative300.bin')
# word_model = tf.convert_to_tensor(data)

word_model = np.load(('wordVectors.npy'))

# tf.reset_default_graph()

# word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.txt')

print('加载word2vec')



input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength],name = 'inputs_reviews')

labels = tf.placeholder(tf.int32, [None, numClass], name = 'labels')
keep_probs = tf.placeholder(tf.float32, name='keep_probs')



# weight = tf.Variable(tf.truncated_normal([lstmUnits, numClass]))

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([ numDimensions, lstmUnits]), name = 'w_in'),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([lstmUnits, numClass]), name = 'w_out')
}


# bias = tf.Variable(tf.constant(0.1, shape=[numClass]))

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[lstmUnits, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[numClass, ]))
}

# W = tf.Variable(tf.constant(0.0, shape=[400000,numDimensions]),
#                     trainable=False, name="W")
# embedding_placeholder = tf.placeholder(tf.float32, [400000 , numDimensions])
# embedding_init = W.assign(embedding_placeholder)
#
# session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# sess = tf.Session(config=session_conf)
# sess.run(embedding_init, feed_dict={embedding_placeholder: word_model})

#
# def lstm(input_x, input_y):
#
#     input_x = tf.placeholder(tf.int32, [None, maxSeqLength], name="input_x")    # X - The Data
#     input_y = tf.placeholder(tf.float32, [None, numClass], name="input_y")      # Y - The Lables
#     dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout
#
#     with tf.device('/cpu:0'), tf.name_scope("embedding"):
#             W = tf.Variable(tf.random_uniform([40000, numDimensions], -1.0, 1.0),name="W")
#             embedded_chars = tf.nn.embedding_lookup(W, input_x)
#
#     lstm_cell = tf.contrib.rnn.LSTMCell(lstmUnits, state_is_tuple=True)
#     lstm_out,lstm_state = tf.nn.dynamic_rnn(lstm_cell, embedded_chars, dtype=tf.float32)
#     val2 = tf.transpose(lstm_out, [1, 0, 2])
#     last = tf.gather(val2, int(val2.get_shape()[0]) - 1)
#     out_weight = tf.Variable(tf.random_normal([lstmUnits, numClass]))
#     out_bias = tf.Variable(tf.random_normal([numClass]))
#
#
#
#             #lstm_final_output = val[-1]
#             #embed()
#     scores = tf.nn.xw_plus_b(last, out_weight,out_bias, name="scores")
#     predictions = tf.nn.softmax(scores, name="predictions")
#
#     losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=input_y)
#     loss = tf.reduce_mean(losses, name="loss")
#
#     correct_pred = tf.equal(tf.argmax(predictions, 1),tf.argmax(input_y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"),name="accuracy")
#     return predictions





def lstm(input_data, weights, biases):

    embed = embeding_layer(vocab_size, numDimensions, input_data)
    data = tf.reshape(embed, (-1, numDimensions))
    # embed = tf.reshape(embed,[-1, numDimensions])
    x_in = tf.matmul(data, weights['in'])+biases['in']

    x_in = tf.reshape(x_in, [-1, maxSeqLength,lstmUnits ])

    cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    init_state = cell.zero_state(batchSize, dtype = tf.float32)


    outputs, final_state = tf.nn.dynamic_rnn(cell, x_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    result = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return result

# with tf.device('/cpu:0'), tf.name_scope("embedding"):
    # W = tf.Variable(
    #     tf.random_uniform([3000000, 300], -1.0, 1.0),
    #     name="W")


#
# W = tf.Variable(tf.constant(0.0, shape=[maxSeqLength, numDimensions]),
#                 trainable=False, name="W")
#
# embedding_placeholder = tf.placeholder(tf.float32, [maxSeqLength, numDimensions])
# embedding_init = W.assign(embedding_placeholder)
#
# ...
# sess = tf.Session()
#
# sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})


pred = lstm(input_data,  weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




# data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.nn.embedding_lookup(word_model, input_data)
#



# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
# value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#
#
#
#
# bias = tf.Variable(tf.constant(0.1, shape=[numClass]))
# value = tf.transpose(value, [1, 0, 2])
# last = tf.gather(value, int(value.get_shape()[0]) - 1)
# prediction =(tf.matmul(last, weight) + bias)
#
#
# correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#
#
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
# optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
#


saver = tf.train.Saver()
tf.add_to_collection('lstm01network', pred)




sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())



tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# with tf.Session() as sess:
#     # tf.initialize_all_variables() no long valid from
#     # 2017-03-02 if using tensorflow >= 0.12
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         init = tf.initialize_all_variables()
#     else:
#         init = tf.global_variables_initializer()
#     sess.run(init)
#     step = 0
#
#     while step * batchSize < iterations:
#         batch_xs, batch_ys = getTrainBatch()
#         # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
#         sess.run([train_op], feed_dict={
#             input_data: batch_xs,
#             labels: batch_ys,
#         })
#         if (step % 50 == 0):
#             summary = sess.run(merged, {input_data: batch_xs, labels: batch_ys})
#             writer.add_summary(summary, step)
#         if step % 20 == 0:
#             print('损失值',sess.run(loss, feed_dict={
#             input_data: batch_xs,
#             labels: batch_ys,
#             }))
#             print('正确率',sess.run(accuracy, feed_dict={
#             input_data: batch_xs,
#             labels: batch_ys,
#             }))
#         step += 1
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(train_op, {input_data: nextBatch, labels: nextBatchLabels})




   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 50== 0 and i != 0):
       # save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print('正确率',sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}))
       print('****损失值',sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels}))
   if (i % 1000 == 0):
        # 保存checkpoint, 同时也默认导出一个meta_graph
        # graph名为'my-model-{global_step}.meta'.
        saver.save(sess, 'model/my-model', global_step=i)

writer.close()
