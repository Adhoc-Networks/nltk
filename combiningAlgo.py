#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:10:51 2019

@author: jeetu
"""
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode (votes)
    
    def confidence (self, features):
        
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count (mode(votes))
        conf =choice_votes / len(votes)
        return conf
#    
#short_pos=open ("short_reviews/positive.txt","r").read()
#short_neg=open ("short_reviews/negative.txt","r").read()
#
#documents=[]
#
#for r in short_pos.split('/n'):
#    documents.append(   (r,"pos")   )
#    
#for r in short_neg.split('/n'):
#    documents.append(   (r,"neg")   )
#
#
#all_words=[]
#short_pos_words=word_tokenize(short_pos)
#short_neg_words=word_tokenize(short_neg)
#
#for w in sort_pos_words:
#    all_words.append(w.lower())
#    
#for w in sort_neg_words:
#    all_words.append(w.lower())
#


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

random.shuffle(featuresets)

#positive data set example
training_set =featuresets[:1900]
testing_set = featuresets[1900:]

#negative data set example
training_set =featuresets[100:]
testing_set = featuresets[:100]


#classifier=nltk.NaiveBayesClassifier.train(training_set)
classifier_file=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_file)
classifier_file.close()

print ("Original Naive Bayes Algo accuracy in percentage:- ",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)


MNB_Classifier=SklearnClassifier(MultinomialNB())
MNB_Classifier.train(testing_set)
print ("MNB_Classifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(MNB_Classifier,testing_set))*100)

#BernoulliNB
BNB_Classifier=SklearnClassifier(BernoulliNB())
BNB_Classifier.train(testing_set)
print ("BernoulliNB Algo accuracy in percentage:- ",(nltk.classify.accuracy(BNB_Classifier,testing_set))*100)

#LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC

LogisticRegression_Classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(testing_set)
print ("LogisticRegression_Classifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(LogisticRegression_Classifier,testing_set))*100)


SGDClassifier=SklearnClassifier(SGDClassifier())
SGDClassifier.train(testing_set)
print ("SGDClassifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(SGDClassifier,testing_set))*100)


#SVC_Classifier=SklearnClassifier(SVC())
#SVC_Classifier.train(testing_set)
#print ("SVC_Classifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(SVC_Classifier,testing_set))*100)


LinearSVC_Classifier=SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(testing_set)
print ("LinearSVC_Classifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(LinearSVC_Classifier,testing_set))*100)


NuSVC_Classifier=SklearnClassifier(NuSVC())
NuSVC_Classifier.train(testing_set)
print ("NuSVC_Classifier Algo accuracy in percentage:- ",(nltk.classify.accuracy(NuSVC_Classifier,testing_set))*100)

########   VOTE
voted_classifier = VoteClassifier(classifier,
                                 MNB_Classifier,
                                 BNB_Classifier,
                                 LogisticRegression_Classifier,
                                 SGDClassifier,
                                 LinearSVC_Classifier,
                                 NuSVC_Classifier)

print ("voted_classifier accuracy percent: " , (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print ("Classification: ", voted_classifier.classify(testing_set[0][0]), "confidence %: " ,voted_classifier.confidence(testing_set[0][0])*100)
print ("Classification: ", voted_classifier.classify(testing_set[1][0]), "confidence %: " ,voted_classifier.confidence(testing_set[1][0])*100)
print ("Classification: ", voted_classifier.classify(testing_set[2][0]), "confidence %: " ,voted_classifier.confidence(testing_set[2][0])*100)
print ("Classification: ", voted_classifier.classify(testing_set[3][0]), "confidence %: " ,voted_classifier.confidence(testing_set[3][0])*100)
print ("Classification: ", voted_classifier.classify(testing_set[4][0]), "confidence %: " ,voted_classifier.confidence(testing_set[4][0])*100)
print ("Classification: ", voted_classifier.classify(testing_set[5][0]), "confidence %: " ,voted_classifier.confidence(testing_set[5][0])*100)
