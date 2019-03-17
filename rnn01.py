
import tensorflow as tf

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=3) 
    y_test = keras.utils.to_categorical(y_test,num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print 'Defining a Simple Keras Model...'
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=3
    model.add(Activation('softmax'))

    print 'Compiling the Model...'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print "Train..." # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

    print "Evaluate..."
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('../model/lstm.h5')
    print 'Test score:', score


def lstm_predict(string):
    print 'loading model......'
    with open('../model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print 'loading weights......'
    model.load_weights('../model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    # print result # [[1]]
    if result[0]==1:
        print string,' positive'
    elif result[0]==0:
        print string,' neutral'
    else:
        print string,' negative'
