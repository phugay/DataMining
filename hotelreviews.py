# simple linear regression 

from __future__ import division
import os, os.path
import operator
import nltk
import re
import math
import sys
import itertools
import string
import re
import numpy as np
import pandas as pd
import operator
from collections import Counter
nltk.download('popular')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from collections import Counter
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn import metrics 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk import PorterStemmer as ps



#words = []
#remove = []
frequency = {}
class_names = {'truthful': 0, 'deceptive' : 1}
ytrain_labels = []
training_data = []
test_data = []
count = 0
counttest = 0
wnl = WordNetLemmatizer()
fname = []
final_tfidf = []
fnamelist = []
sortedlist = []

#step 1 : working on training data
#calling function training
training(count)

#step 2: working on test data
testing(counttest)

#accessing the training data followed by calling all the necassary function.
def training(count) :
    print('Working on training data..')
    for root, dirs, files in os.walk('/Users/pranjalihugay/Documents/testmining/Training'):
        #for r in root:
        for name in files:
            dir_class = (root+'/'+name).split('/')[-2]
            if dir_class in list(class_names.keys()):
                fname = name
                #print(fname)
                ytrain_labels.append(class_names[dir_class])
                #y_labels=np.asarray(ytrain_labels)
                count+=1
                a=open(root+'/'+name,'r',encoding="ISO-8859-1").read(); 
                wrds = removespecialchars(a)
                #print(wrds)
                tokens = createtokens(wrds)
                #print(tokens)
                filtertokens = [w for w in tokens if not w.lower() in stopwords.words('english')]
                #print(filtertokens)
                clean = cleanme(filtertokens)
                clean = ' '.join(clean)
                training_data.append(clean)
                #print(training_data)
    print('no of training files:')
    print(count)
    #split into training and validation
    print('splitting into train and validation data using cross validation..')
    X_train, X_valid, y_train, y_valid = train_test_split(training_data, ytrain_labels, test_size=0.33, random_state=42)
    #calling function to find tf idf values
    print('doing tfidf evaluation..')
    X_train, X_valid = find_TFIDF(X_train,X_valid)
    #last step is to call prediction function where knn has been implemented
    print('predicting result of training using knn implementation..')
    prediction(X_valid,X_train)
    

def testing(counttest): 
    print('Working on test data..')
    for root, dirs, files in os.walk('/Users/pranjalihugay/Documents/testmining/Testing'):
        #for r in root:
        for name in files:
            dir_class = (root+'/'+name).split('/')[-2]
            fname = name
            fname = fname.split('.')
            int_fname = (int)(fname[0])
            fnamelist.append(int_fname)
            sortedlist = sorted(fnamelist)
            counttest+=1        
    for item in sortedlist:
        itemstr = str(item)
        constr = itemstr+'.txt'
        a=open('/Users/pranjalihugay/Documents/testmining/Testing/'+constr,'r',encoding="ISO-8859-1").read(); 
        wrds = removespecialchars(a)
        tokens = createtokens(wrds)
        filtertokens = [w for w in tokens if not w.lower() in stopwords.words('english')]
        clean = cleanme(filtertokens)
        clean = ' '.join(clean)
        test_data.append(clean)
    print('no of testing files:')        
    print(counttest)
    print('doing tfidf evaluation..')
    X_train,X_test = find_TFIDF_test(training_data,test_data)
    print('predicting result of test data using knn implementation..')
    prediction_test(X_test,X_train)


def find_TFIDF(X_train, X_valid):
    vector = TfidfVectorizer(analyzer='word', input='content', stop_words=stopwords.words('english'), ngram_range=(2,2))
    train_tfidf = vector.fit_transform(X_train)
    valid_tfidf = vector.transform(X_valid)
    return train_tfidf.toarray(), valid_tfidf.toarray()


def find_TFIDF_test(training_data,test_data):
    vector = TfidfVectorizer(analyzer='word', input='content', stop_words=stopwords.words('english'),ngram_range=(2,2))
    train_tfidf = vector.fit_transform(training_data)
    test_tfidf = vector.transform(test_data)
    return train_tfidf.toarray(), test_tfidf.toarray()


#prediction function for train and validation data
def prediction(X_valid, X_train):
    y_predict = []
    for v in X_valid:
        ct = -1
        sortedtvdistance = []
        y_outcome = []
        y_frequent = []
        cosine_result = []
        trainvaliddist = []
        for t in X_train:
            cosine_result = 1 - spatial.distance.cosine(v, t)
            ct+=1
            trainvaliddist.append((abs(cosine_result),ct))
        sortedtvdistance=sorted(trainvaliddist, key=lambda tup: tup[0],reverse = True)[:91] #k-value of 5
        for stvd in sortedtvdistance:
             y_outcome.append(y_train[stvd[1]])
        y_frequent = Counter(y_outcome).most_common()
        y_predict.append(y_frequent[0][0])     
    print(y_predict)
    acc = metrics.accuracy_score(y_valid, y_predict)
    print(acc*100)


#prediction function for test data
def prediction_test(X_test,X_train):
    y_predict = []
    for v in X_test:
        ct = -1
        sortedtvdistance = []
        y_outcome = []
        y_frequent = []
        cosine_result = []
        trainvaliddist = []
        for t in X_train:
            cosine_result = 1 - spatial.distance.cosine(v, t)
            ct+=1
            trainvaliddist.append((abs(cosine_result),ct))
        sortedtvdistance=sorted(trainvaliddist, key=lambda tup: tup[0],reverse = True)[:91]
        for stvd in sortedtvdistance:
            y_outcome.append(ytrain_labels[stvd[1]])
        y_frequent = Counter(y_outcome).most_common()
        y_predict.append(y_frequent[0][0])     
    print(y_predict)
    with open('/Users/pranjalihugay/Documents/CS584_MINING/bstoutput.txt', 'w') as f:
        for item in y_predict:
            f.write("%s\n" % item)

#preprocessing functions
def cleanme(ft):
    sent = [x.lower() for x in ft]
    wrds1 = [wnl.lemmatize(wrd) for wrd in sent]
    clwrds = [w for w in wrds1 if not w in stopwords.words('english')]
    return(clwrds)               
        
    
def removespecialchars(f):
    remove = re.sub('[^\w\s]','',f)
    remove = re.sub('_','',remove)
    remove = re.sub('\s+',' ',remove)
    remove = re.sub(',', '',remove)
    remove = ''.join(c for c in remove if c not in punctuation)
    remove = remove.strip()
    return remove


def createtokens(words):
    tokens = nltk.word_tokenize(words)
    return tokens


