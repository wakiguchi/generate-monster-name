import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
tf.set_random_seed(1234)

#Neural Network
class DNN(object):
    def __init__(self, n_in, n_hiddens, n_out):
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self._x = None
        self._t = None,
        self._keep_prob = None
        self._sess = None
        self._history = {
            'accuracy': [],
            'argmax': [],
            'loss': []
        }

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def inference(self, x, keep_prob):
        # Input - hidden, hidden - hidden
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i-1]

            self.weights.append(self.weight_variable([input_dim, n_hidden]))
            self.biases.append(self.bias_variable([n_hidden]))

            h = tf.nn.relu(tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)

        # hidden - output
        self.weights.append(
            self.weight_variable([self.n_hiddens[-1], self.n_out]))
        self.biases.append(self.bias_variable([self.n_out]))

        y = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return y

    def loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y,1e-10,1.0)), axis=1))
        return cross_entropy

    def training(self, loss):
        #Adagrad
        optimizer = tf.train.AdagradOptimizer(0.05)

        #Adam
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)

        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    #BODY of training
    def fit(self, X_train, Y_train,
            nb_epoch=100, batch_size=100, p_keep=0.5,
            verbose=1):
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)

        self._x = x
        self._t = t
        self._keep_prob = keep_prob

        y = self.inference(x, keep_prob)

        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)
        self._accuracy = accuracy

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        # Set saver
        saver = tf.train.Saver()

        # loop num epoch
        for epoch in range(nb_epoch):
            X_, Y_ = shuffle(X_train, Y_train)

            #loop num batchsize
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                sess.run(train_step, feed_dict={
                    x: X_[start:end],
                    t: Y_[start:end],
                    keep_prob: p_keep
                })

            loss_ = loss.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })
            accuracy_ = accuracy.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train,
                keep_prob: 1.0
            })
            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch:', epoch,
                      ' loss:', loss_,
                      ' accuracy:', accuracy_)


        # Save Model
        saver.save(sess, "model/name_model.ckpt")
        return self._history

    def evaluate(self, X_test, Y_test):
        return self._accuracy.eval(session=self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test,
            self._keep_prob: 1.0
        })

    #BODY of prediction
    def forward(self, X_train,p_keep=1.0):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None,self.n_in])
        keep_prob = tf.placeholder(tf.float32)

        self._x = x
        y = self.inference(x,keep_prob)
        am = tf.argmax(y,1)

        sess = tf.Session()

        #Load model
        saver = tf.train.Saver()
        saver.restore(sess, "model/name_model.ckpt")
        self._sess = sess

        am_list = am.eval(session=sess,feed_dict={
            x: X_train,
            keep_prob: p_keep
        })
        sess.close()
        self._history['argmax'].extend(am_list)
        return self._history

#Normalization process
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

#Data processing
def dataprocessing(df):
    '''
    Data processing!
    '''

    # Fill nulls in type2
    df.loc[df.type2.isnull(), 'type2'] = '無'

    # Vectorize the letters of the name.(1-hot vector)
    pokename_vectorizer = DictVectorizer(sparse=False)
    name_list = list(df['name'])
    char_name_all = []
    for each_name in name_list:
        char_name_all.extend(list(each_name))
        # name!
        # '!' is end point of name, so add '!' end of each names.
        char_name_all.append("!")
    d_df = pd.DataFrame(char_name_all)
    d = d_df.to_dict('records')
    # Y_all is 1-hot vector of letters.
    Y_all = pokename_vectorizer.fit_transform(d)
    names = pokename_vectorizer.get_feature_names()

    # Character parameters
    x_bs = np.array(df[['v_h', 'v_a', 'v_b', 'v_c', 'v_d', 'v_s']])
    x_bs = zscore(x_bs)

    # Vectorize pokemon type1
    poketype1_vectorizer = DictVectorizer(sparse=False)
    d = df[['type1']].to_dict('record')
    x_type1 = poketype1_vectorizer.fit_transform(d)

    # Vectorize pokemon type2
    poketype2_vectorizer = DictVectorizer(sparse=False)
    d = df[['type2']].to_dict('record')
    x_type2 = poketype2_vectorizer.fit_transform(d)
    return (name_list,x_type1,x_type2,x_bs,Y_all,char_name_all,names)

if __name__ == '__main__':
    # Set seed
    tf.set_random_seed(0)

    # Load data
    df = pd.read_csv('data/database_pokemon_mini.csv')
    name_list,x_type1,x_type2,x_bs,Y_all,char_name_all,names = dataprocessing(df)

    # Make Input feature
    # Convine [x_bs, x_type1, x_type2, pre_letter ,number] teacher forcing
    X_all = []
    count = 0
    for i in range(len(x_bs)):
        for j in range(len(name_list[i])+1):
            X = []
            X.extend(x_bs[i])
            X.extend(x_type1[i])
            X.extend(x_type2[i])
            if count == 0:
                X.extend(Y_all[5])  # '!' vector
            else:
                X.extend(Y_all[count-1]) # pre_letter vector
            X.append(j) #number of letters of the name
            X_all.append(X)
            count += 1

    #Show the dimentions of Input and Output features
    print('Input dimensions:' + str(len(X_all)))
    print('Output dimensions:' + str(len(Y_all)))

    '''
    Setting model!
    '''
    model = DNN(n_in=len(X_all[0]),n_hiddens=[400, 400, 200],n_out=len(Y_all[0]))

    '''
    Training!
    '''
    nb_epoch=2
    model.fit(X_all, Y_all,
              nb_epoch,
              batch_size=8,
              p_keep=0.5)

    # Custom font
    plt.rc('font', family='serif')
    # prepare graph
    fig = plt.figure()
    # write data to graph
    plt.plot(range(nb_epoch), model._history['loss'], label='loss', color='black')
    # axis name
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # save and show the graph
    plt.show()
    plt.savefig('model/name_model.eps')
