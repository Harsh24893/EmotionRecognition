import os
import sys
import re
import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split

np.random.seed(7)

DIR_GLOVE = 'glove/glove.840B.300d/'
DIR_DATA = 'data/'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def gloveVec(filename):
    embeddings = {}
    f = open(os.path.join(DIR_GLOVE, filename), encoding='utf-8')
    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
        except ValueError:
            i += 1
    f.close()
    return embeddings

def loadData(filename):
    df = pd.read_csv(DIR_DATA + filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    labels = sorted(list(set(df[selected[0]].tolist())))
    dict.fromkeys(set(df[selected[0]].tolist()))
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i

    x_train = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_train = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    y_train = to_categorical(np.asarray(y_train))
    return x_train,y_train

def createVocabAndData(sentences):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    vocab = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return vocab,data


def createEmbeddingMatrix(word_index,embeddings_index):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def lstmModel(embedding_matrix,epoch):
    model = Sequential()
    n, embedding_dims = embedding_matrix.shape

    model.add(Embedding(n, embedding_dims, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(LSTM(128, dropout=0.6, recurrent_dropout=0.6))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(X_train, y_train, validation_split=VALIDATION_SPLIT, epochs=epoch, batch_size=128)
    model.save_weights('text_lstm_weights.h5')

    scores= model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == "__main__":

    sentences, labels = loadData('iseardataset.csv')
    embeddings = gloveVec('glove.840B.300d.txt')
    vocab, data = createVocabAndData(sentences)
    embedding_mat = createEmbeddingMatrix(vocab,embeddings)
    pickle.dump([data, labels, embedding_mat], open('embedding_matrix.pkl', 'wb'))
    print ("Data created")

    print("Train Test split")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SPLIT, random_state=42)

    lstmModel(embedding_mat,40)



