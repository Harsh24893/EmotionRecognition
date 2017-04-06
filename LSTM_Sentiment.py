__author__ = 'Hao'

__author__ = 'Hao'

from keras.datasets import imdb
import numpy as np
import re
import itertools
from collections import Counter
import cPickle
import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from utils import expand_label
from sklearn.metrics import accuracy_score, roc_auc_score
from data_prepare import transform_text

class TextLSTM:
    def __init__(self, embedding_mat=None, maxlen=56,
                 nb_lstm=128, n_vocab=10000, embedding_dims=300):
        """
        :param embedding_mat: The embedding word2vec matrix, of size(n_vocab, embedding_dims)
                when it's None, using random generated
        :param maxlen: the max length of a given sentence
        :param filter_length: filter length in convolution layer
        :param nb_filters: number of filters
        :param filter_length:
        :return:
        """
        if embedding_mat is not None:
            self.n_vocab, self.embedding_dims = embedding_mat.shape
        else:
            self.n_vocab = n_vocab
            self.embedding_dims = embedding_dims
        self.maxlen = maxlen
        self.nb_lstm=128

        print "Building the model"
        self.model=Sequential()

        #Model embedding layer, for word index-> word embedding transformation
        if embedding_mat is not None:
            self.model.add(Embedding(self.n_vocab, self.embedding_dims,
                            weights=[embedding_mat], input_length=self.maxlen))
        else:
            self.model.add(Embedding(self.n_vocab, self.embedding_dims, input_length=self.maxlen))

        self.model.add(LSTM(self.nb_lstm))  # try using a GRU instead, for fun
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')


    def fit(self, X_train, y_train, X_test, y_test,
            batch_size=100, nb_epoch=3, show_accuracy=True):
        """
        :param X_train: each instance is a list of word index
        :param y_train:
        :return:
        """
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        y_train = expand_label(y_train)
        y_test = expand_label(y_test)

        self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(X_test, y_test))


    def predict(self, X_test):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        return self.model.predict_classes(X_test)

    def save_weights(self, fname='../data/text_lstm_weights.h5'):
        self.model.save_weights(fname)

    def load_model(self, fname='../data/text_lstm_weights.h5'):
        self.model.load_weights(fname)

    def predict_prob(self, X_test):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        return self.model.predict_proba(X_test)

    def accuracy_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def auc_score(self, X_test, y_test):
        y_pred = self.predict_prob(X_test)[:,1]
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
    print "Loading the data"
    x, y, embedding_mat = cPickle.load(open('../data/train_mat.pkl'))
    vocab = cPickle.load(open('../data/vocab.pkl'))

    print "Train Test split"
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

    print "Training"
    clf = TextLSTM(embedding_mat=embedding_mat)
    # clf.fit(X_train, y_train, X_test, y_test, nb_epoch=4)
    # clf.save_weights()

    print "Loading"
    clf.load_model()
    texts = [
        "I like you.",
        "I am extremely angry",
        "I feel sorry...",
        "I studied in NYU."
    ]
    l = clf.predict_text(texts, vocab)
    print l[:,1]


    print "Evaluation on test set"
    print "Accuracy: %.3f" %clf.accuracy_score(X_test, y_test)
    print "AUC: %.3f" %clf.auc_score(X_test, y_test)
