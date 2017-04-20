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
    #token_list = stemming_token_list(token_list)
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
    Data = pd.read_csv('iseardataset.txt',header=None)
    dict = {}
    #Data = pd.read_csv('text_emotion.csv',header=None)
    #Data = pd.read_csv('preprocessed_yelp.csv',header=None)
    #print Data[2]
    #print len(Data[1])
    for i in range (len(Data[0])):
##        if i < 10:
##            print Data[2][i]+' '+Data[0][i]
        #line = Data[2][i]+'|'+Data[0][i]
        if Data[0][i] not in dict.keys():
            dict[Data[0][i]]=1
        else:
            dict[Data[0][i]]+=1
        
        line = Data[0][i]+'|'+cleanDataset(Data[1][i])
        #line = Data[1][i]+'|'+Data[3][i]
        #print line
        list.append(line)
    print dict
    random.shuffle(list)
    c = 0
    f = open('isear.txt','w')
    for i in range(int(len(list))):
        #print list[i]
        if c < 10:
            #print list[i][4:]
            #print list[i]
            c = c+ 1        
        index = list[i].index('|')
        f.write(list[i][:index]+'\t'+list[i][index+1:])
        f.write('\n')
        train_data.append(list[i][index+1:])
        train_labels.append(list[i][:index])
    f.close()

