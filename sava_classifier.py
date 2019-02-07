#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:19:02 2019

@author: jeetu
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC
#from sklearn.exceptions import NotFittedError
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

#this is similar to this method
'''

documents=[]

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append(list(movie_reviews.word(fileid)),category)
        
 '''       
random.shuffle(documents)
#print (documents)

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)

word_features=list (all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features ={}
    for w in word_features:
        features[w]=(w in words)
    
    return features
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents] 

training_set =featuresets[:1900]
testing_set = featuresets[1900:]

#classifier=nltk.NaiveBayesClassifier.train(training_set)
classifier_file=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_file)
classifier_file.close()

print ("Naive Bayes Algo accuracy in percentage:- ",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier=open("naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()