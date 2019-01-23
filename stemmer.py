#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:16:39 2019

@author: jeetu
"""
#importing stemming library
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps=PorterStemmer()
example_words=["python","pythoned","pythoning","pythoner","pythonly"]

#stemming the example words
#for w in example_words:
#    print (ps.stem(w))
new_text="It is very important to be pythonly while you are pyhtoning with python. All pythoners have pythoned poorly at least once."

words=word_tokenize(new_text)

for w in words:
    print(ps.stem(w))