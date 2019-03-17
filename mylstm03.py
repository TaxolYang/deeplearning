import tensorflow as tf
import numpy as np
import gensim
from tensorflow.contrib.tensorboard.plugins import projector


from random import randint

import datetime


train_vec = np.load('train_index.npy')
train_label = np.load('train_label.npy')
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


tf.reset_default_graph()
lr = 0.001
maxSeqLength = 29 #Maximum length of sentence
numDimensions = 50 #Dimensions for each word vector
# wordVectors = np.load(('wordVectors.npy'))
batchSize = 64
lstmUnits = 256
numClass = 3
iterations = 100000
# word_model = gensim.models.KeyedVectors.load_word2vec_format('D:/data/vec/google/GoogleNews-vectors-negative300.bin', binary=True)

# word_model = gensim.models.word2vec.Word2Vec.load('D:/data/vec/google/GoogleNews-vectors-negative300.bin')
# word_model = tf.convert_to_tensor(data)

word_model = np.load(('wordVectors.npy'))
print(word_model.shape)

tf.reset_default_graph()


# word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.txt')

print('加载word2vec')


labels = tf.placeholder(tf.int32, [batchSize, numClass])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# with tf.device('/cpu:0'), tf.name_scope("embedding"):
#     W = tf.Variable(
#         tf.random_uniform([3000000, 300], -1.0, 1.0),
#         name="W")



# W = tf.Variable(tf.constant(0.0, shape=[maxSeqLength, numDimensions]),
#                 trainable=False, name="W")

# embedding_placeholder = tf.placeholder(tf.float32, [maxSeqLength, numDimensions])
# embedding_init = W.assign(embedding_placeholder)

# ...
# sess = tf.Session()

# sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_model, input_data)




lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)




weight = tf.Variable(tf.truncated_normal([lstmUnits, numClass]))
bias = tf.Variable(tf.constant(0.1, shape=[numClass]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction =(tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)







sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())



tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

average_loss = 0
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})




   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 50== 0 and i != 0):
       # save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print(sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels}))
       print('****损失值',sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels}))
writer.close()
