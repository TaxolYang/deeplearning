import tensorflow as tf
import numpy as np
import re
import gensim
from collections import Counter


path = 'D:\\data\\2017\\2017_English_final\\GOLD\\Subtask_A\\twitter-2016train-A.txt'

wordsList =np.load('wordsList.npy')
print('Loaded the word list')
wordsList = wordsList.tolist()
wordsList = [Word.decode('UTF-8')for Word in wordsList]
wordsVectors = np.load(('wordVectors.npy'))
print('Loaded the word vectors!')

numwords = []


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower()
    string = string.strip('@')
    return re.sub(strip_special_chars, "", string.lower())


def calword(contents):
    # print('加载word2vec模型')
    # word_model = gensim.models.KeyedVectors.load_word2vec_format('D:/data/vec/google/GoogleNews-vectors-negative300.bin', binary=True)



    # print('加载word2vec模型完成')



    max = 29
    ids = np.zeros((len(contents), max), dtype='int32')


    print('开始存储索引')
    indexCounter = 0
    sentence_num = 0

    for text_tokens in contents:
        # print(text_tokens)
        # counters = len(text_tokens.split())
        # numwords.append(counters)
        cleanedLine = cleanSentences(text_tokens)

        word_list = text_tokens.split()


        text_token = []


        # print(word_list)
        for i, word in enumerate(word_list):
                cleanedLine = cleanSentences(word)
                # print(cleanedLine)
                # print('第', i, '个词语是', word)

                try:
                    # 将词转换为索引index
                    # print(word)
                    # text_token.append(word_model.vocab[word].index)
                    ids[sentence_num][i] = wordsList.index(cleanedLine)
                    # word_list[i] = word_model[word]

                #print('*'*10,cut_list[i], '#'*10,  cn_model.vocab[word], '*'*10, cn_model.vocab[word].count, len(cn_model[word]), word)
                except ValueError:
                    # 如果词不在字典中，则输出0
                    ids[sentence_num][i] = 0
                i = i + 1
                if i >= max:
                    break

        # text_list[sentence_num].append(text_token)
        sentence_num = sentence_num+1




    #     num_tokens = [len(tokens) for tokens in text_list]
    #     num_tokens = np.array(num_tokens)
    # # print(text_list)
    #
    # max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)  # 表示句子的长度
    # max_tokens = int(max_tokens)
    # print("我要设定的最大索引矩阵的长度", max_tokens)
    # #
    # #     # 取tokens的长度为236时，大约95%的样本被涵盖
    # #     # 我们对长度不足的进行padding，超长的进行修剪
    # #     # sum统计
    # sum = 0
    # for i in range(len(num_tokens)):
    #     if num_tokens[i] < max_tokens:
    #         sum = sum + 1
    #         rate = sum / len(num_tokens)
    # print("在设定的最大索引矩阵的长度", max_tokens, "时，能完整的表示的评论数字为", rate)
    return ids



# def max_word(matirx):
#
#     for text_tokens in contents:
#         # print(text_tokens)
#         counters = len(text_tokens.split())
#         numwords.append(counters)
#         word_list = text_tokens.split()
#
#         # for i, word in enumerate(word_list):
#                 # print(word)
#                 #     word_list[i] = word
#
#                 # try:
#                 #     # 将词转换为索引index
#                 #     word_list[i] = word_model[word]
#                 # #                print('*'*10,cut_list[i], '#'*10,  cn_model.vocab[word], '*'*10, cn_model.vocab[word].count, len(cn_model[word]), word)
#                 # except KeyError:
#                 #     # 如果词不在字典中，则输出0
#                 #     word_list[i] = 0
#
#         text_list.append(word_list)
#         num_tokens = [len(tokens) for tokens in text_list]
#         num_tokens = np.array(num_tokens)
#     # print(text_list)
#
#     max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)  # 表示句子的长度
#     max_tokens = int(max_tokens)
#     print("我要设定的最大索引矩阵的长度", max_tokens)
#     #
#     #     # 取tokens的长度为236时，大约95%的样本被涵盖
#     #     # 我们对长度不足的进行padding，超长的进行修剪
#     #     # sum统计
#     sum = 0
#     for i in range(len(num_tokens)):
#         if num_tokens[i] < max_tokens:
#             sum = sum + 1
#             rate = sum / len(num_tokens)
#     print("在设定的最大索引矩阵的长度", max_tokens, "时，能完整的表示的评论数字为", rate)
#

    # return max_tokens

num, labels, contents, = [], [], []

with open(path ,'r', encoding='utf-8') as f:
    # try:
    #     while True:
    #         line = f.readline()
    #     # print(lines_list)
    #     # for line in lines_list:
    #     #     print(line)
    #         num, label, content = line.strip().split('\t')
    #             # print(num)
    #             # print(label)
    #             # print(content)
    #
    #
    #
    #         contents.append(content)
    #         labels.append(label)
    # except KeyError:
    #     pass
    with open(path ,'r', encoding='utf-8') as f:
     for line in f:
        try:
            num, label, content = line.strip().split('\t')
            # print(num)
            # print(label)
            # print(content)



            contents.append(content)
            labels.append(label)
        except:
                pass
    print('读取文件完成')


def change_label(labels):
    new_labels = []
    print(labels)
    for i in labels:
        if i =='negative':
            new_labels.append(-1)
        elif i=='neutral':
            new_labels.append(0)
        elif i == 'positive':
            new_labels.append(1)


    # print(new_labels)
    b = tf.one_hot(new_labels,3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        labels = sess.run(b)

    return labels

# print(contents)
# print(len(contents))
# print(calword(contents))

matirx =calword(contents)
# #
# # matirx =change_label(labels)
#
# print(matirx)
# np.save('train_index', matirx)

# NUM_CLASSES = 10 # 10分类
# label = [0,1,2,3] # sample label
# batch_size = tf.size(labels) # get size of labels : 4
# label = tf.expand_dims(labels, 1) # 增加一个维度
# indices = tf.expand_dims(tf.range(0, batch_size,1), 1) #生成索引
# concated = tf.concat([indices, label] , 1) #作为拼接



# index=[0,1,2,3]
# one_hot=tf.one_hot(index,5)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     print(sess.run(one_hot))

new_labels = []
# print(labels)



# new_labels =   change_label(labels)
# len(new_labels)
# print(new_labels)
# len(contents)


# train_vec = np.zeros((len(contents),300),dtype='float32')
# np.save('train_vec',test_matrix)
# print(np.load('train_vec.npy'))
def save_matrix3(matrix,str):
    len(matrix)


    new_matrix = np.zeros((len(matrix),),dtype='int32')
    np.save(str, new_matrix)
    return

def save_matrix300(matrix,str):
    len(matrix)


    newmatrix = np.zeros((len(matrix),300),dtype='float32')
    np.save(str, newmatrix)

    return

# test_matrix = calword(contents)
# # test_matrix = change_label(labels)
# # print(max_word(contents))
#
#
# # save_matrix3(test_matrix, 'train_index')
# save_matrix3(test_matrix, 'train_index')
