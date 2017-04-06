import numpy as np
import re
import itertools
import pickle
from collections import Counter
from keras.optimizers import RMSprop, Adadelta
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten,Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from utils import expand_label
from sklearn.metrics import accuracy_score, roc_auc_score
from data_prepare import transform_text

class modelCNN:
    def __init__(self, embedding_mat=None, maxlen=56, filter_length=[3, 4, 5],nb_filters=300, n_vocab=10000, embedding_dims=300):

        if embedding_mat is not None:
            self.n_vocab, self.embedding_dims = embedding_mat.shape
        else:
            self.n_vocab = n_vocab
            self.embedding_dims = embedding_dims

        self.maxlen = maxlen
        self.filter_length = filter_length
        self.nb_filters = nb_filters

        self.model = Graph()
        self.model.add_input('input', input_shape=(self.maxlen,), dtype='int')

        if embedding_mat is not None:
            self.model.add_node(Embedding(self.n_vocab, self.embedding_dims, weights=[embedding_mat], input_length=self.maxlen),name='embedding', input='input')
        else:
            self.model.add_node(Embedding(self.n_vocab, self.embedding_dims, input_length=self.maxlen), name='embedding', input='input')

        conv_layer = []
        nb_filter_each = nb_filters / len(filter_length)
        for each_length in filter_length:
            self.model.add_node(Convolution1D(
                nb_filter=nb_filter_each,
                filter_length=each_length,
                border_mode='valid',
                activation='relu',
                subsample_length=1),
                name=str(each_length) + '_convolution',
                input='embedding')
            self.model.add_node(MaxPooling1D(pool_length=(self.maxlen - each_length) / 1 + 1),
                                name=str(each_length) + '_pooling',
                                input=str(each_length) + '_convolution')
            self.model.add_node(Flatten(), name=str(each_length) + '_flatten', input=str(each_length) + '_pooling')
            conv_layer.append(str(each_length) + '_flatten')

        self.model.add_node(Dropout(0.5), name='dropout', inputs=conv_layer)
        self.model.add_node(Dense(2, W_regularizer=l2(0.01)), name='full_connect', input='dropout')
        self.model.add_node(Activation('softmax'), name='softmax', input='full_connect')

        adadelta = Adadelta(lr=0.95, rho=0.95, epsilon=1e-6)
        self.model.add_output(name='output', input='softmax')

        self.model.compile(loss={'output': 'binary_crossentropy'}, optimizer=adadelta)

    def fit(self, X_train, y_train, X_test, y_test, batch_size=50, nb_epoch=3):

        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        y_train = expand_label(y_train)
        y_test = expand_label(y_test)

        # early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        self.model.fit({'input': X_train, 'output': y_train}, batch_size=batch_size, nb_epoch=nb_epoch,
                       verbose=1, validation_data=({'input': X_test, 'output': y_test}), callbacks=[early_stop])

    def save_weights(self, fname='../data/text_cnn_weights.h5'):
        self.model.save_weights(fname)

    def load_model(self, fname='../data/text_cnn_weights.h5'):
        self.model.load_weights(fname)

    def predict(self, X_test):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        X_pred = np.array(self.model.predict({'input': X_test})['output'])
        X_pred = np.argmax(X_pred, axis=1)
        return X_pred

    def predict_prob(self, X_test):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        X_pred = np.array(self.model.predict({'input': X_test})['output'])
        return X_pred

    def accuracy_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def auc_score(self, X_test, y_test):
        y_pred = self.predict_prob(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred)

    def predict_text(self, X_text, vocab):
        """
        Get a list of texts and make predictions directly
        :param X_text:
        :param vocab:
        :return:
        """
        x = [transform_text(i, vocab) for i in X_text]
        return self.predict_prob(x)

if __name__ == "__main__":
    print("Data loading")
    x, y, embedding_mat = pickle.load(open('../data/train_mat.pkl'))

    print("Split Train Test")
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

    print("Training")
    clf = modelCNN(embedding_mat=embedding_mat)
    clf.fit(X_train, y_train, X_test, y_test, nb_epoch=10)
    # clf.load_model(fname='../data/cnn_diff_filter.h5')

    print("Dumping the model")
    clf.save_weights(fname='../data/cnn_diff_filter.h5')

    print("Evaluation on test set")
    print("Accuracy: %.3f" %clf.accuracy_score(X_test, y_test))
    print("AUC: %.3f" %clf.auc_score(X_test, y_test))