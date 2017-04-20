from tabulate import tabulate
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit

# TRAIN_SET_PATH = "20ng-no-stop.txt"
# TRAIN_SET_PATH = "r52-all-terms.txt"
TRAIN_SET_PATH = "isear.txt"

GLOVE_6B_50D_PATH = "glove.6b\glove.6B.300d.txt"
GLOVE_840B_300D_PATH = "glove.840B.300d\glove.840B.300d.txt"

i = 0

X, y = [], []
with open(TRAIN_SET_PATH, "rb") as infile:
    for line in infile:
        label, text = line.split("\t")
##        if i < 10:
##            print label,'---', text
##            i = i + 1
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(text.split())
        y.append(label)
##list = []
##Data = pd.read_csv('iseardataset.txt',header=None)
##for i in range (len(Data[0])):
####    if i < 10:
####        print Data[2][i]+' '+Data[0][i]
##    line = Data[0][i]+'|'+Data[1][i]
##    list.append(line)
##random.shuffle(list)
##c = 0
##for i in range(int(len(list)*0.7)):
##    
##    index = list[i].index('|')
##    if c < 10:
##        print list[i][index+1:],'---',list[:index]
##        c = c+ 1
##    X.append(list[i][index+1:].split())
##    y.append(list[i][:index])

X, y = np.array(X), np.array(y)
##print y[:10]
##print X[:10]
print "total examples %s" % len(y)

import numpy as np
with open(GLOVE_6B_50D_PATH, "rb") as lines:
    word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}

# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have 
# enough RAM - remove the 'if' line and load everything

glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove_small[word] = np.array(nums)
print 'Glove Data set read'
glove_big = {}
with open(GLOVE_840B_300D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove_big[word] = np.array(nums)

# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
print 'Making the model'
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
model.wv.index2word
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
print 'Pipeline'
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
##

print 'Class vectorizer'
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
##
# Extra Trees classifier is almost universally great, let's stack it with our embeddings

print 'Etree'
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=50))])

##
##
all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
##    ("bern_nb", bern_nb),
##    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("glove_small", etree_glove_small), 
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("glove_big", etree_glove_big), 
    ("glove_big_tfidf", etree_glove_big),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
]

scores = sorted([(name, cross_val_score(model, X, y, cv=3).mean()) 
                 for name, model in all_models], 
                key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
##
plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])

def benchmark(model, X, y, n):
    #print n,' ',y
    test_size = 1 - (n / float(len(y)))
    print 'Test Split', test_size
    scores = []
    #print len(y),'pagal hpgyaa hoon',' ',test_size
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #print 'Idhar dekho kya aata hai...'
        #print len(y_train), len(y_test)
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)

#train_sizes = [10, 40, 160, 640, 3200, 6400]
#train_sizes = [ 3200]
train_sizes = [5261, 6012, 6764]
table = []
for name, model in all_models:
    print '~~~~~~~~~~~~~~~~~~~~MODEL: ',name,' ~~~~~~~~~~~~~~~~~~~~~~'
    for n in train_sizes:
        
        accuracy = benchmark(model, X, y, n)
        print 'ACCURACY : ',accuracy
        table.append({'model': name, 
                      'accuracy': accuracy, 
                      'train_size': n})
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print ''
    print ''
df = pd.DataFrame(table)


##
##plt.figure(figsize=(15, 6))
##fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
##                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "w2v_tfidf", 
##                                                         "glove_small_tfidf", "glove_big_tfidf", 
##                                                        ])])
##sns.set_context("notebook", font_scale=1.5)
##fig.set(ylabel="accuracy")
##fig.set(xlabel="labeled training examples")
##fig.set(title="R8 benchmark")
##fig.set(ylabel="accuracy")
