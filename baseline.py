# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn


import sys
import os
import time
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import random
import unicodedata
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

NLTK_STOPWORDS = set(stopwords.words('english'))
# In[2]:
def usage():
    print('Usage:')
    print('python %s <data_dir>' % sys.argv[0])

def lowercase( s):
        return s.lower()
    
def tokenize( s):
    token_list = nltk.word_tokenize(s)
    return token_list

def remove_punctuation( s):
    return s.translate(None, string.punctuation)

def remove_numbers( s):
    return s.translate(None, string.digits)

def remove_stopwords( token_list):
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return filter(exclude_stopwords, token_list)

def stemming_token_list( token_list):
    STEMMER = PorterStemmer()
    
    #print token_list.decode('utf-8')
    return [STEMMER.stem(tok.decode('utf-8')) for tok in token_list]

def restring_tokens( token_list):
    return ' '.join(token_list)

# Function to clean the reviews using the Pre-Processing functions written above
def cleanDataset( line):
    cleanData = ''
    line = lowercase(line)
    printable = set(string.printable)
    line = filter(lambda x: x in printable, line)
    #line = unicodedata.normalize('NFKD', line).encode('ascii','ignore')
    line = remove_punctuation(line)
    line = remove_numbers(line)
    token_list = tokenize(line)
    token_list = remove_stopwords(token_list)
    token_list = stemming_token_list(token_list)
    for words in token_list:
        cleanData+=words+' '
    return cleanData


if __name__ == '__main__':
    print 'Entered'
    if len(sys.argv) > 2:
        usage()
        sys.exit(1)
    print 'Entered 1'
    data_dir = 'txt_sentoken'
    classes = ['pos', 'neg']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    list = []
    Data = pd.read_csv('Data/iseardataset.csv',header=None)
    #Data = pd.read_csv('text_emotion.csv',header=None)
    #Data = pd.read_csv('preprocessed_yelp.csv',header=None)
    #print Data[2]
    #print len(Data[1])
    for i in range (len(Data[0])):
##        if i < 10:
##            print Data[2][i]+' '+Data[0][i]
        #line = Data[2][i]+'|'+Data[0][i]
        line = Data[0][i]+'|'+Data[1][i]
        #line = Data[1][i]+'|'+Data[3][i]
        list.append(line)
##    f = open('combined.txt','w')
##    f1 = open('pos','r')
##    c = 0
##    for i in f1:
##        i = cleanDataset(i)
##        line = 'pos|'+i
##        f.write(line)
##        list.append(line)
##        f.write('\n')
##    f1 = open('neg','r')
##    c = 0
##    for i in f1:
##        i = cleanDataset(i)
##        line = 'neg|'+i
##        f.write(line)
##        list.append(line)
##        f.write('\n')
##    f.close()
    random.shuffle(list)
    c = 0
    for i in range(int(len(list)*0.7)):
        if c < 10:
            #print list[i][4:]
            #print list[i]
            c = c+ 1
        index = list[i].index('|')
        train_data.append(list[i][index+1:])
        train_labels.append(list[i][:index])
    
    
    for i in range(int(len(list)*0.7)+1, len(list)):
        index = list[i].index('|')
        test_data.append(list[i][index+1:])
        test_labels.append(list[i][:index])    
##    for curr_class in classes:
##        dirname = os.path.join(data_dir, curr_class)
##        for fname in os.listdir(dirname):
##            with open(os.path.join(dirname, fname), 'r') as f:
##                content = f.read()
##                if fname.startswith('cv9'):
##                    test_data.append(content)
##                    test_labels.append(curr_class)
##                else:
##                    train_data.append(content)
##                    train_labels.append(curr_class)

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    print len(prediction_rbf),' ', len(test_labels)
    c = 0
    for i in range(len(test_labels)):
        if prediction_rbf[i]==test_labels[i]:
            c += 1
                       
        print prediction_rbf[i],' ', test_labels[i]
    print 'ACCURACY RBF= ',float((c*1.0)/len(test_labels))
    

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    print len(prediction_linear),' ', len(test_labels)
    c = 0
    for i in range(len(test_labels)):
        if prediction_linear[i]==test_labels[i]:
            c += 1
                       
        print prediction_linear[i],' ', test_labels[i]
    print 'ACCURACY LINEAR= ',float((c*1.0)/len(test_labels))

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    print len(prediction_liblinear),' ', len(test_labels)
    c = 0
    for i in range(len(test_labels)):
        if prediction_liblinear[i]==test_labels[i]:
            c += 1
                       
        print prediction_liblinear[i],' ', test_labels[i]
    print 'ACCURACY LIBLINEAR= ',float((c*1.0)/len(test_labels))

    # Print results in a nice table
    print('Results for SVC(kernel=rbf)')
    print('Training time: %fs; Prediction time: %fs' % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print('Results for SVC(kernel=linear)')
    print('Training time: %fs; Prediction time: %fs' % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print('Results for LinearSVC()')
    print('Training time: %fs; Prediction time: %fs' % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))

    print ''
    print 'Bhai ab aagye Random Forest Classifier finally'
    print train_vectors.shape, test_vectors.shape
    classifier = RandomForestClassifier().fit(train_vectors.toarray(), train_labels)
    print '......'
    prediction_labels = classifier.predict(test_vectors.toarray())
    c = 0
    for i in range(len(test_labels)):
        if prediction_labels[i]==test_labels[i]:
            c += 1
                       
        print prediction_labels[i],' ', test_labels[i]
    print 'ACCURACY Random Forest= ',float((c*1.0)/len(test_labels))
    print('Results for Random Forest Classifier')
    print(classification_report(test_labels, prediction_labels))
    
