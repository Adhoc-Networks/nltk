#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:26:01 2019

@author: jeetu
"""
#importing corpora for stopword
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#data
example_sentence="This is an example showing off stop word function."

#english stopwords
stop_words= set(stopwords.words("english"))
#word tokenizing of data
words= word_tokenize(example_sentence)
#
#filtered_sent=[]
#for w in words:
#    if w not in stop_words:
#        filtered_sent.append(w)
#    
#print (filtered_sent)

#storing the filtered data in a list
filtered_sent=[w for w in words if not w in stop_words]
print(filtered_sent)