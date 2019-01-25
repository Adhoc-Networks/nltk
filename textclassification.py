#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:37:30 2019

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
print(all_words.most_common(15))
print(all_words["good"])