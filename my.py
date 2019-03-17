import tensorflow as tf
#生成一个先入先出队列和一个Queuerunner，生成文件名队列

filenames = ['D:/data/2017/new/2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义 decoder
num, label, sent = tf.decode_csv(value, record_defaults=[['null'], ['null'], ['null']], field_delim=' ')#['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
example_batch, label_batch = tf.train.batch([label, sent], batch_size=1, capacity=200, num_threads=2)#保证样本和标签一一对应
#运行图
with tf.Session() as sess:
    coord = tf.train.Coordinator()#创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队
    for i in range(100):
        e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)
    coord.request_stop()
    coord.join(threads)
