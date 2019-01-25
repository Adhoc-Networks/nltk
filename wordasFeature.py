#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:16:52 2019

@author: jeetu
"""

import nltk
import random
from nltk.corpus import movie_reviews

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
        features[w]={w in words}
    
    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents] 